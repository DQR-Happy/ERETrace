"""Restricted PTX control-flow certification, not an arbitrary-PTX verifier.

Integer control is interpreted exactly at a launch's scalar specialization.
Data loads are TOP; bounded remainder/bit masks recover a finite value set.
Unknown predicates fork; a repeated abstract state or exhausted exploration
budget rejects the certificate, never returns the longest path seen so far.
Float/shuffle values are TOP, so their influence on control is conservative.
"""
import hashlib,itertools,json,re
from dataclasses import dataclass

class Unsupported(ValueError): pass
TOP=None

def val(v):return frozenset([v])
def finite(v):return TOP if v is None or len(v)>64 else frozenset(v)
def cast(x,t):
    if t=='pred':return int(bool(x))
    bits=int(t[1:]);x&=(1<<bits)-1
    return x-(1<<bits) if t[0]=='s' and x>>(bits-1) else x

def split_args(s):return re.split(r',\s*(?![^{}]*\})',s)

@dataclass
class Ins:
    pred:str;op:str;args:list;line:int;site:int=-1

class Program:
    def __init__(self,ptx):
        self.sha256=hashlib.sha256(ptx.encode()).hexdigest()
        s=re.sub(r'/\*.*?\*/|//[^\n]*','',ptx,flags=re.S)
        if len(re.findall(r'\.entry\s',s))!=1:raise Unsupported('one entry required')
        m=re.search(r'\.entry\s+(\S+)\s*\((.*?)\)\s*\{',s,re.S)
        if not m:raise Unsupported('entry declaration')
        self.name=m[1];self.params=[]
        for p in m[2].split(','):
            q=re.search(r'\.(?:u|s|b|f)\d+\s+(\w+)(?:\[(\d+)\])?',p)
            if q:self.params.append(q[1])
        self.code=[];self.labels={};self.sites=[]
        # Nested lexical blocks can shadow register names; this frontend does
        # not perform name resolution and must reject them, not flatten them.
        closed=False
        for ln,line in enumerate(s[m.end():].splitlines(),1):
            line=line.strip()
            if not line:continue
            if closed:raise Unsupported('content after entry body')
            if line=='{':raise Unsupported('nested register scopes are unsupported')
            if line=='}':closed=True;continue
            if line.startswith('.'):
                if not line.startswith(('.reg ','.pragma ')):raise Unsupported('body directive '+line)
                continue
            if line.endswith(':'):
                if line[:-1] in self.labels:raise Unsupported('duplicate label')
                self.labels[line[:-1]]=len(self.code);continue
            z=re.fullmatch(r'(@!?%[\w$]+\s+)?([\w.]+)(?:\s+(.*?))?;',line)
            if not z:raise Unsupported('instruction must occupy one line: '+line)
            ins=Ins((z[1] or '').strip().lstrip('@'),z[2],split_args(z[3] or ''),ln)
            base=ins.op.split('.')[0]
            allowed={'ld','st','mov','cvta','cvt','add','sub','mul','mad','neg','not','and','or','xor','shl','shr','div','rem','setp','selp','bra','ret','exit','bar','shfl','fma','rsqrt','ex2','lg2','sqrt','abs','min','max'}
            if base not in allowed:raise Unsupported('unsupported opcode '+ins.op)
            if base=='bar' and (ins.op!='bar.sync' or len(ins.args)!=1):raise Unsupported('only full-CTA bar.sync supported')
            if base=='bra' and ins.op not in ['bra','bra.uni']:raise Unsupported('branch modifier')
            if base in ['ret','exit'] and ins.op not in ['ret','ret.uni','exit']:raise Unsupported('terminal modifier')
            if base=='shfl' and ins.op!='shfl.sync.bfly.b32':raise Unsupported('shuffle variant')
            # Never approximate an unimplemented integer modifier (e.g. add.sat)
            # using the semantics of its unmodified mnemonic.
            integer_ops=r'(?:mov|neg|not|and|or|xor|shl|shr|add|sub|min|max|abs|div|rem)\.(?:[usb](?:8|16|32|64)|pred)|mul\.(?:lo|hi|wide)\.[us](?:16|32|64)|mad\.lo\.[us](?:16|32|64)|cvt\.[usb](?:8|16|32|64)\.[usb](?:8|16|32|64)|cvta(?:\.to)?\.(?:global|shared)\.u64|setp\.(?:eq|ne|lt|le|gt|ge)\.[us](?:16|32|64)|selp\.[usb](?:16|32|64)'
            if base not in ['ld','st','bra','ret','exit','bar','shfl'] and '.f' not in ins.op and not re.fullmatch(integer_ops,ins.op):raise Unsupported('unsupported integer semantics '+ins.op)
            if base in ['ld','st'] and not any('.'+space+'.' in ins.op for space in ['param','global','shared']):raise Unsupported('memory space '+ins.op)
            if base in ['ld','st'] and not re.match(r'^(?:ld|st)\.(?:param|global|shared)\.',ins.op):raise Unsupported('memory ordering/volatile modifier')
            if '.param.' in ins.op and (base!='ld' or '.v' in ins.op):raise Unsupported('only scalar parameter loads supported')
            if ins.op.startswith(('ld.global.','st.global.')):
                b=re.search(r'\.[busf](8|16|32|64)$',ins.op)
                addr=next((x for x in ins.args if '[' in x),None)
                if not b or not addr or not re.fullmatch(r'\[%[\w$]+(?:[+-](?:0x[\da-fA-F]+|\d+))?\]',addr):raise Unsupported('memory operands')
                ins.site=len(self.sites);self.sites.append(dict(site=ins.site,opcode=ins.op,width=int(b[1])//8*(int(re.search(r'\.v([24])\.',ins.op)[1]) if '.v' in ins.op else 1)))
            self.code.append(ins)
        if not closed:raise Unsupported('unterminated entry body')
        for i in self.code:
            if i.op.startswith('bra') and (len(i.args)!=1 or i.args[0] not in self.labels):raise Unsupported('indirect/missing branch')
        if not self.sites:raise Unsupported('no global sites')
        needed={i.pred.lstrip('!') for i in self.code if i.pred}
        while True:
            before=len(needed)
            for i in self.code:
                dst=set(re.findall(r'%[\w$]+',i.args[0]))
                if dst&needed and not i.op.startswith(('ld.global','ld.shared','shfl')) and '.f' not in i.op:
                    needed.update(re.findall(r'%[\w$]+(?:\.[xyz])?',','.join(i.args[1:])))
            if len(needed)==before:break
        self.control_regs=needed

    def walk(self,params,grid,block,cta,tid,budget=100000,witness=False):
        regs={}
        for prefix,values in [('tid',tid),('ntid',block),('ctaid',cta),('nctaid',grid)]:
            regs.update({'%'+prefix+'.'+axis:val(v) for axis,v in zip('xyz',values)})
        def get(s,r):
            if s in r:return r[s]
            if s=='WARP_SZ':return val(32)
            try:return val(int(s,0))
            except ValueError:return TOP
        def predicate(p,r):
            if not p:return val(1)
            v=get(p.lstrip('!'),r)
            if v is None:return frozenset([0,1])
            return frozenset(1-int(bool(x)) if p.startswith('!') else int(bool(x)) for x in v)
        def evaluate(ins,r):
            op=ins.op;a=ins.args;base=op.split('.')[0];types=re.findall(r'(?:^|\.)([usb]\d+|pred)(?=\.|$)',op);typ=types[-1] if types else None
            if base=='ld':
                if '.param.' in op:
                    address=a[1][1:-1];v=params.get(address,TOP);return v
                return TOP
            if base in ['shfl','rsqrt','ex2','lg2','sqrt','fma'] or '.f' in op:return TOP
            if base=='mov' or base=='cvta':return get(a[1],r)
            if base=='cvt':
                if len(types)!=2:return TOP
                v=get(a[1],r);return finite({cast(cast(x,types[1]),types[0]) for x in v}) if v is not None else TOP
            vs=[get(x,r) for x in a[1:]]
            if base=='setp':
                if len(vs)!=2:raise Unsupported('compound predicate comparison')
                if any(v is None for v in vs):return frozenset([0,1])
                cmp=op.split('.')[1];fs={'eq':lambda x,y:x==y,'ne':lambda x,y:x!=y,'lt':lambda x,y:x<y,'le':lambda x,y:x<=y,'gt':lambda x,y:x>y,'ge':lambda x,y:x>=y}
                if cmp not in fs or typ is None:raise Unsupported('comparison '+op)
                return frozenset(int(fs[cmp](cast(x,typ),cast(y,typ))) for x,y in itertools.product(*vs))
            if base=='selp':
                cond=vs[2] if vs[2] is not None else [0,1];sets=[vs[0] if c else vs[1] for c in cond]
                return TOP if any(v is None for v in sets) else finite(set().union(*sets))
            if any(v is None for v in vs):
                # Sound finite images for unknown bit patterns / unsigned remainder.
                if base=='and' and len(vs)==2 and vs[1] is not None and len(vs[1])==1:
                    mask=next(iter(vs[1]))
                    if 0<=mask<64:return frozenset(x for x in range(mask+1) if x&~mask==0)
                if base=='rem' and typ and typ[0]=='u' and vs[1] is not None and len(vs[1])==1:
                    m=next(iter(vs[1]));return frozenset(range(m)) if 0<m<=64 else TOP
                return TOP
            if typ is None:raise Unsupported('integer type '+op)
            def calc(v):
                v=[cast(x,typ) for x in v];x=v[0];y=v[1] if len(v)>1 else 0;bits=1 if typ=='pred' else int(typ[1:])
                if base=='add':z=x+y
                elif base=='sub':z=x-y
                elif base in ['mul','mad']:
                    z=x*y
                    if '.hi.' in op:z>>=bits
                    if base=='mad':z+=v[2]
                    if '.wide.' in op:return cast(z,typ[0]+str(bits*2))
                elif base=='neg':z=-x
                elif base=='not':z=not x if typ=='pred' else ~x
                elif base=='and':z=x&y
                elif base=='or':z=x|y
                elif base=='xor':z=x^y
                elif base in ['shl','shr']:
                    y=min(cast(y,'u32'),bits);z=x<<y if base=='shl' else x>>y
                elif base in ['div','rem']:
                    if y==0:raise Unsupported('possible division by zero')
                    q=abs(x)//abs(y)*(-1 if (x<0)!=(y<0) else 1);z=q if base=='div' else x-q*y
                elif base=='min':z=min(x,y)
                elif base=='max':z=max(x,y)
                elif base=='abs':z=abs(x)
                else:raise Unsupported('integer transfer '+op)
                return cast(z,typ)
            return finite({calc(v) for v in itertools.product(*vs)})
        # Iterative DFS avoids recursion limits on legitimate long counted loops.
        stack=[(0,regs,0,[],frozenset())];maximum=0;paths=0;steps=0;one=[]
        while stack:
            pc,r,n,events,seen=stack.pop()
            while True:
                steps+=1
                if steps>budget:raise Unsupported('exploration budget exhausted; no bound certified')
                if pc>=len(self.code):raise Unsupported('control falls through entry')
                state=(pc,tuple(sorted(r.items())))
                if state in seen:raise Unsupported('recurrent abstract state / unresolved loop')
                seen=seen|{state};ins=self.code[pc];choices=predicate(ins.pred,r)
                if 0 in choices:
                    if 1 in choices:stack.append((pc+1,dict(r),n,list(events),seen))
                    else:pc+=1;continue
                base=ins.op.split('.')[0]
                if ins.site>=0:
                    n+=1
                    if n>65536:raise Unsupported('per-thread capacity exceeds supported bound')
                    if witness:
                        address=next(x for x in ins.args if '[' in x)[1:-1];m=re.fullmatch(r'(%[\w$]+)(?:([+-])(0x[\da-fA-F]+|\d+))?',address);v=get(m[1],r)
                        if v is None or len(v)!=1:raise Unsupported('address witness is not concrete')
                        ptr=next(iter(v))+(int(m[3],0)*(1 if m[2]=='+' else -1) if m[2] else 0)
                        events.append((ins.site,ptr&((1<<64)-1),self.sites[ins.site]['width']))
                if base in ['ret','exit']:
                    maximum=max(maximum,n);paths+=1;one=events;break
                if base=='bra':pc=self.labels[ins.args[0]];continue
                if base not in ['st','bar'] and (witness or set(re.findall(r'%[\w$]+',ins.args[0]))&self.control_regs):
                    v=evaluate(ins,r)
                    for dst in re.findall(r'%[\w$]+',ins.args[0]):r[dst]=v if '|' not in ins.args[0] else TOP
                pc+=1
        if witness and paths!=1:raise Unsupported('nondeterministic witness')
        return dict(bound=maximum,states=steps,paths=paths,events=one if witness else [])

def specialize(ptx,params,grid,block):
    p=Program(ptx);grid=tuple(grid);block=tuple(block)
    if len(grid)!=3 or len(block)!=3 or any(not isinstance(v,int) or v<1 for v in (*grid,*block)) or __import__('math').prod(block)>1024 or __import__('math').prod(grid)*__import__('math').prod(block)>=2**31:raise Unsupported('launch dimensions')
    vals={k:(TOP if v is None else frozenset(v) if isinstance(v,list) else val(v)) for k,v in params.items()}
    caps=[];states=paths=0;cache={}
    for cz in range(grid[2]):
      for cy in range(grid[1]):
       for cx in range(grid[0]):
        for tz in range(block[2]):
         for ty in range(block[1]):
          for tx in range(block[0]):
           builtins=dict(zip(['%ctaid.x','%ctaid.y','%ctaid.z','%tid.x','%tid.y','%tid.z'],[cx,cy,cz,tx,ty,tz]))
           key=tuple((k,v) for k,v in builtins.items() if k in p.control_regs)
           if key not in cache:cache[key]=p.walk(vals,grid,block,(cx,cy,cz),(tx,ty,tz));states+=cache[key]['states'];paths+=cache[key]['paths']
           caps.append(cache[key]['bound'])
    return dict(schema=1,analyzer_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
      ptx_sha256=p.sha256,entry=p.name,params=params,grid=list(grid),block=list(block),capacities=caps,sites=p.sites,
      abstract_states=states,terminal_paths=paths,control_classes=len(cache),slots=sum(caps),maximum=max(caps),complete=True)

def check_certificate(ptx,cert,params,grid,block):
    if cert.get('ptx_sha256')!=hashlib.sha256(ptx.encode()).hexdigest() or cert.get('params')!=params or cert.get('grid')!=list(grid) or cert.get('block')!=list(block):raise Unsupported('certificate identity mismatch')
    expected=specialize(ptx,params,grid,block)
    if cert!=expected:raise Unsupported('certificate does not reproduce')
    return True
