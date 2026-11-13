"""GPU oracle for bounded single-pass memory events and predicated exits."""
import argparse, ctypes as C, hashlib, json, struct, subprocess
from pathlib import Path
import numpy as np
from neutrino.probe.core_events import instrument_memory_events
from neutrino.probe.cuda import probing
from neutrino.common import Probe, Map, Register

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--arch',default='sm_86');p.add_argument('--negative-address',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
cu=C.CDLL('libcuda.so.1');ptr=C.c_uint64;handle=C.c_void_p
def bind(n,args):
    f=getattr(cu,n);f.argtypes=args;f.restype=C.c_int;return f
def ok(v):
    if v:raise RuntimeError(f'CUDA {v}')
init=bind('cuInit',[C.c_uint]);dg=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(handle),C.c_int]);current=bind('cuCtxSetCurrent',[handle])
alloc=bind('cuMemAlloc_v2',[C.POINTER(ptr),C.c_size_t]);free=bind('cuMemFree_v2',[ptr]);htod=bind('cuMemcpyHtoD_v2',[ptr,C.c_void_p,C.c_size_t]);dtoh=bind('cuMemcpyDtoH_v2',[C.c_void_p,ptr,C.c_size_t]);load=bind('cuModuleLoad',[C.POINTER(handle),C.c_char_p]);get=bind('cuModuleGetFunction',[C.POINTER(handle),handle,C.c_char_p]);launch=bind('cuLaunchKernel',[handle,*([C.c_uint]*7),handle,C.POINTER(handle),C.POINTER(handle)])
streamcreate=bind('cuStreamCreate',[C.POINTER(handle),C.c_uint]);streamsync=bind('cuStreamSynchronize',[handle]);streamdestroy=bind('cuStreamDestroy_v2',[handle])
ok(init(0));dev=C.c_int();ctx=handle();ok(dg(C.byref(dev),0));ok(retain(C.byref(ctx),dev));ok(current(ctx))
modules=[]
def compile_ptx(name,code):
    path=a.output/(name+'.ptx');path.write_text(code);r=subprocess.run(['ptxas','-arch='+a.arch,'-v',str(path),'-o',str(path.with_suffix('.cubin'))],text=True,capture_output=True);path.with_suffix('.log').write_text(r.stderr);r.check_returncode()
    module=handle();fun=handle();ok(load(C.byref(module),str(path.with_suffix('.cubin')).encode()));ok(get(C.byref(fun),module,b'events'));modules.append(module);return fun
def mem(n):
    d=ptr();ok(alloc(C.byref(d),n));return d
def write(d,data):ok(htod(d,data,len(data)))
def read(d,n):
    b=C.create_string_buffer(n);ok(dtoh(b,d,n));return b.raw
def call(f,grid,block,args,stream=None):
    ar=(handle*len(args))(*(C.addressof(x) for x in args));ok(launch(f,grid,1,1,block,1,1,0,stream,ar,None))
ptx='''.version 8.4
.target ARCH
.address_size 64
.visible .entry events(
.param .u64 data
)
{
.reg .b64 %base, %addr, %off;
.reg .b32 %t, %cta, %nt, %id, %v, %j, %bit;
.reg .pred %odd, %early, %four, %again;
ld.param.u64 %base, [data];
mov.u32 %t, %tid.x;
mov.u32 %cta, %ctaid.x;
mov.u32 %nt, %ntid.x;
mad.lo.u32 %id, %cta, %nt, %t;
mul.wide.u32 %off, %id, 64;
add.u64 %addr, %base, %off;
ld.global.u32 %v, [%addr];
and.b32 %bit, %t, 1;
setp.ne.u32 %odd, %bit, 0;
and.b32 %bit, %t, 3;
setp.eq.u32 %early, %bit, 1;
mov.u32 %j, 0;
LOOP:
@%odd ld.global.u32 %v, [%addr+4];
@%odd add.u32 %v, %v, 1;
@%odd st.global.u32 [%addr+4], %v;
add.u32 %j, %j, 1;
setp.eq.u32 %four, %j, 4;
and.pred %four, %four, %early;
@%four ret;
setp.lt.u32 %again, %j, 8;
@%again bra LOOP;
st.global.u32 [%addr+8], %v;
exit;
}'''.replace('ARCH',a.arch)
if a.negative_address:
    ptx=ptx.replace('add.u64 %addr, %base, %off;', 'add.u64 %addr, %base, %off;\nadd.u64 %addr, %addr, 16;').replace('[%addr+4]', '[%addr+-12]').replace('[%addr+8]', '[%addr+-8]').replace('[%addr]', '[%addr+-16]')
native=compile_ptx('native',ptx);cases=[]
for cap in [4,32]:
    code,sites=instrument_memory_events(ptx,cap);f=compile_ptx('events-cap'+str(cap),code)
    for block in [1,31,32,33,48,64,127]:
        grid=3;n=grid*block;data=mem(n*64);trace=mem(n*cap*32+128);counts=mem(n*8+128)
        initial=np.arange(n*16,dtype=np.uint32).tobytes();write(data,initial);call(native,grid,block,[data]);expected=read(data,n*64)
        for stream_mode in ['default','independent']:
            stream=handle()
            if stream_mode=='independent':ok(streamcreate(C.byref(stream),1))
            write(data,initial);write(trace,b'\xa5'*(n*cap*32+128));write(counts,b'\xa5'*(n*8+128))
            call(f,grid,block,[data,trace,counts],stream);ok(streamsync(stream))
            actual=read(data,n*64);assert actual==expected,'stateful kernel replayed or modified'
            raw=read(trace,n*cap*32+128);cnt=read(counts,n*8+128);assert raw[-128:]==b'\xa5'*128 and cnt[-128:]==b'\xa5'*128
            attempted=np.frombuffer(cnt[:n*8],dtype='<u8');checked=0;overflows=0;masks=set()
            for i in range(n):
                t=i%block;odd=t%2;early=t%4==1;iterations=4 if early else 8
                sequence=[0]+([1,2]*iterations if odd else [])+([] if early else [3])
                assert attempted[i]==len(sequence),(cap,block,i,int(attempted[i]),sequence)
                overflows+=len(sequence)>cap
                for j,site in enumerate(sequence[:cap]):
                    tm,addr,sid,tid,mask,width=struct.unpack_from('<QQIIII',raw,(i*cap+j)*32)
                    offset=[0,4,4,8][site]
                    assert tm and addr==data.value+i*64+offset and sid==site and tid==i and width==4
                    assert mask&(1<<(t%32)),(t,mask)
                    if site in (1,2):assert not(mask&0x55555555),'inactive even lane recorded'
                    masks.add(mask);checked+=1
                unwritten=raw[(i*cap+min(len(sequence),cap))*32:(i+1)*cap*32]
                assert unwritten==b'\xa5'*len(unwritten)
            cases.append(dict(capacity=cap,block=block,grid=grid,stream=stream_mode,checked_records=checked,attempted=int(attempted.sum()),overflow_threads=overflows,complete=overflows==0,allocated_bytes=n*(cap*32+8),masks=sorted(masks),model_digest=hashlib.sha256(actual).hexdigest()))
            if stream_mode=='independent':ok(streamdestroy(stream))
        for d in [data,trace,counts]:ok(free(d))

# Fixed warp records: positive/negative predicates and exit share one terminal
# record per lane 0, while non-lane-0 threads finish at a different point.
maps=[Map('T','warp','array',16,1,[Register('start','u64'),Register('elapsed','u32'),Register('sm','u32')])]
probe=Probe('sched','warp',['kernel'],'mov.u64 %NR0, %clock64;',
    'mov.u64 %NR1, %clock64;\nsub.u64 %NR1, %NR1, %NR0;\nmov.u64 %NR2, 123;\nSAVE [T] {%NR0, %NR1, %NR2};')
terminal=ptx.replace('@%four ret;','@%four ret;').replace('exit;','@!%four exit;\nret;')
for mode in ['corrected','shift','late','vector','late_vector']:
    f=compile_ptx('terminal-'+mode,probing(terminal,[probe],maps,3,mode));n=3*48;data=mem(n*64);trace=mem(6*16+128);write(data,bytes(n*64));write(trace,b'\xa5'*(6*16+128));call(f,3,48,[data,trace]);raw=read(trace,6*16+128);assert raw[-128:]==b'\xa5'*128
    for i in range(6):
        st,el,tag=struct.unpack_from('<QII',raw,i*16);assert st and el and tag==123
    cases.append(dict(terminal_mode=mode,records=6,guard_ok=True))
    ok(free(data));ok(free(trace))
(a.output/'summary.json').write_text(json.dumps(dict(passed=True,sites=sites,cases=cases),indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(cases))))
