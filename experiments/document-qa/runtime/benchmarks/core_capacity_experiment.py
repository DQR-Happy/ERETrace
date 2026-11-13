"""Complete bounded-event comparison, including table cost and data-dependent loops."""
import argparse,ctypes as C,hashlib,json,math,random,struct,subprocess,time
from pathlib import Path
import numpy as np
from core_driver import Driver,P,H
from neutrino.probe.cuda import COMMON_BUFFER_CALC
from neutrino.probe.core_events import instrument_memory_events
from neutrino.probe.core_capacity import capacity_classes,private_layout,instrument_stratified_events

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--round',type=int,default=1)
p.add_argument('--arch',default='sm_86');p.add_argument('--build',action='store_true');p.add_argument('--smoke',action='store_true')
p.add_argument('--three-d',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);build=a.output/'build';build.mkdir(exist_ok=True);root=Path(__file__).resolve().parents[1]

def fixture(limit,dynamic):
    return f'''.version 8.4
.target {a.arch}
.address_size 64
.visible .entry events(.param .u64 data)
{{
{COMMON_BUFFER_CALC.replace('%buf','%model_index')}
.reg .u64 %base, %addr, %off;
.reg .u32 %v, %j, %bit, %limit;
.reg .pred %odd, %early, %four, %again;
ld.param.u64 %base, [data];
mul.wide.u32 %off, %model_index1, 64;
add.u64 %addr, %base, %off;
ld.global.u32 %v, [%addr];
{f'rem.u32 %limit, %v, {limit};' if dynamic else f'mov.u32 %limit, {limit-1};'}
add.u32 %limit, %limit, 1;
and.b32 %bit, %model_index15, 1;
setp.ne.u32 %odd, %bit, 0;
and.b32 %bit, %model_index15, 3;
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
setp.lt.u32 %again, %j, %limit;
@%again bra LOOP;
st.global.u32 [%addr+8], %v;
exit;
}}'''

def plan(limit,dynamic):
    # Safe abstract event counts: unknown data-dependent paths take a maximum.
    early=('events',1+2*min(4,limit)+(1 if limit<4 else 0))
    if dynamic:
        early=('unknown',early,('events',2+2*min(3,limit)))
    body=('residue',4,[1],early,('residue',2,[1],('events',2+2*limit),('events',2)))
    return body,capacity_classes(body)

def compile_one(tag,code):
    pt=build/(tag+'.ptx');pt.write_text(code)
    start=time.perf_counter_ns()
    r=subprocess.run(['ptxas','-arch='+a.arch,'-v',str(pt),'-o',str(pt.with_suffix('.cubin'))],text=True,capture_output=True)
    pt.with_suffix('.log').write_text(r.stderr);r.check_returncode()
    return time.perf_counter_ns()-start

shapes=[(8,False,3,33),(8,True,3,33)] if a.smoke else [(l,d,82,b) for l,d in [(8,False),(8,True),(32,True)] for b in [33,256]]
if a.three_d:shapes=[(8,False,12,33),(8,True,12,48),(32,True,12,40),(8,True,12,64)]
compile_rows=[]
if a.build:
    subprocess.run(['g++','-O3','-shared','-fPIC',str(root/'benchmarks/capacity_repeat.cpp'),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64/stubs','-lcuda','-o',str(build/'repeat.so')],check=True)
    for limit,dynamic in sorted({(l,d) for l,d,_,_ in shapes}):
        src=fixture(limit,dynamic);ast,caps=plan(limit,dynamic)
        for mode in ['N','U','P','T','O']:
            if mode=='N':code=src
            elif mode=='U':code,_=instrument_memory_events(src,max(caps))
            else:code,_=instrument_stratified_events(src,[max(0,c-1) for c in caps] if mode=='O' else caps,'table' if mode=='T' else 'periodic')
            tag=f'l{limit}-d{int(dynamic)}-{mode}'
            ns=compile_one(tag,code);compile_rows.append(dict(tag=tag,compile_ns=ns,code_sha256=hashlib.sha256(code.encode()).hexdigest(),caps=caps,description=ast))
    (a.output/'compilation.json').write_text(json.dumps(compile_rows,indent=2)+'\n')

driver=Driver();repeat=C.CDLL(str(build/'repeat.so')).capacity_repeat
repeat.argtypes=[H,C.c_int,C.c_int,P,P,P,P,C.c_int,C.c_int,C.POINTER(C.c_float)];repeat.restype=C.c_int
rows=[];checks=[];rng=random.Random(31042+a.round)
dtype=np.dtype([('time','<u8'),('address','<u8'),('site','<u4'),('thread','<u4'),('mask','<u4'),('width','<u4')])

def validate(records,counts,offsets,caps,n,block,limit,dynamic,initial,data_ptr):
    ids=np.arange(n);local=ids%block;odd=local%2==1
    lim=(initial[:,0]%limit+1).astype(np.int64) if dynamic else np.full(n,limit)
    early=(local%4==1)&(lim>=4);iters=np.where(early,4,lim)
    expected=np.where(odd,1+2*iters+(~early),2)
    actual=np.frombuffer(counts[:n*8],dtype='<u8');assert np.array_equal(actual,expected)
    retained=np.minimum(expected,caps);owners=np.repeat(ids,retained)
    js=np.concatenate([np.arange(c,dtype=np.int64) for c in retained])
    indices=np.repeat(offsets,retained)+js
    r=np.frombuffer(records[:-128],dtype=dtype)[indices]
    last=(~early[owners])&(js==expected[owners]-1)
    sites=np.where(js==0,0,np.where(last,3,np.where(js%2,1,2)))
    addresses=data_ptr+owners*64+np.choose(sites,[0,4,4,8])
    assert np.all(r['time']>0) and np.array_equal(r['site'],sites)
    assert np.array_equal(r['thread'],owners) and np.array_equal(r['address'],addresses) and np.all(r['width']==4)
    assert np.all((r['mask']>>(local[owners]%32))&1)
    inner=(sites==1)|(sites==2);assert not np.any(r['mask'][inner]&0x55555555)
    assert records[-128:]==b'\xa5'*128 and counts[-128:]==b'\xa5'*128
    return dict(attempted=int(expected.sum()),retained=int(retained.sum()),overflow_threads=int(np.sum(expected>caps)),complete=bool(np.all(expected<=caps)))

for limit,dynamic,grid,block in shapes:
    n=grid*block;ast,caps=plan(limit,dynamic);functions={m:driver.function(build/f'l{limit}-d{int(dynamic)}-{m}.cubin','events') for m in ['N','U','P','T','O']}
    initial=np.arange(n*16,dtype=np.uint32).reshape(n,16);initial[:,0]=(np.arange(n)*37+3+a.round)%1024;raw_input=initial.tobytes()
    gshape=(2,3,2) if a.three_d else (grid,1,1)
    bshape={33:(3,11,1),48:(4,4,3),40:(2,4,5),64:(4,8,2)}[block] if a.three_d else (block,1,1)
    data=driver.memory(len(raw_input));driver.upload(data,raw_input);driver.call(functions['N'],gshape,bshape,[data]);oracle=driver.read(data,len(raw_input))
    order=['N','U','P','T'];rng.shuffle(order)
    for mode in order+['O']:
        start=time.perf_counter_ns()
        budgets=[max(caps)]*len(caps) if mode in ['N','U'] else ([max(0,c-1) for c in caps] if mode=='O' else caps)
        layout,total=private_layout(budgets,block,grid)
        offsets=np.array([r[0] for r in layout],dtype=np.int64);limits=np.array([r[1] for r in layout],dtype=np.int64)
        plan_raw=np.asarray(layout,dtype='<u8').tobytes();planning_ns=time.perf_counter_ns()-start
        trace_bytes=total*32;records=driver.memory(trace_bytes+128);counts=driver.memory(n*8+128);table=driver.memory(len(plan_raw) if mode=='T' else 1)
        host_records=driver.pinned(trace_bytes);host_counts=driver.pinned(n*8)
        if mode=='T':driver.upload(table,plan_raw)
        setup_ns=time.perf_counter_ns()-start
        args=[data,records]+([table] if mode=='T' else [])+[counts]
        driver.upload(data,raw_input);driver.upload(records,b'\xa5'*(trace_bytes+128));driver.upload(counts,b'\xa5'*(n*8+128))
        driver.call(functions[mode],gshape,bshape,args);driver.ok(driver.sync())
        assert driver.read(data,len(raw_input))==oracle
        if mode!='N':
            check=validate(driver.read(records,trace_bytes+128),driver.read(counts,n*8+128),offsets,limits,n,block,limit,dynamic,initial,data.value)
            assert (not check['complete']) if mode=='O' else check['complete']
            checks.append(dict(mode=mode,limit=limit,dynamic=dynamic,grid=grid,block=block,grid_shape=gshape,block_shape=bshape,**check))
        if mode!='O' and not a.three_d:
            samples=[];completion=[];copy_samples=[];loops=30 if a.smoke else 100
            for _ in range(3):
                driver.upload(data,raw_input);ms=C.c_float()
                driver.ok(repeat(functions[mode],grid,block,data,records,counts,table,int(mode=='T'),10,C.byref(ms)))
                driver.ok(repeat(functions[mode],grid,block,data,records,counts,table,int(mode=='T'),loops,C.byref(ms)))
                samples.append(ms.value*1000/loops)
                driver.upload(data,raw_input);t=time.perf_counter_ns();driver.call(functions[mode],(grid,1,1),(block,1,1),args);driver.ok(driver.sync());copy_start=time.perf_counter_ns()
                if mode!='N':
                    driver.ok(driver.dtoh(host_records,records,trace_bytes));driver.ok(driver.dtoh(host_counts,counts,n*8))
                end=time.perf_counter_ns();completion.append(end-t);copy_samples.append(end-copy_start)
            resource=driver.resources(functions[mode],block)
            rows.append(dict(mode=mode,limit=limit,dynamic=dynamic,grid=grid,block=block,capacities=caps,
                kernel_interval_us=float(np.median(samples)),kernel_samples_us=samples,completion_ns=float(np.median(completion)),copy_ns=float(np.median(copy_samples)),
                payload_bytes=trace_bytes if mode!='N' else 0,count_bytes=n*8 if mode!='N' else 0,
                plan_device_bytes=len(plan_raw) if mode=='T' else (8*(2*len(caps)+1) if mode=='P' else 0),
                planning_ns=planning_ns,setup_ns=setup_ns,model_digest=hashlib.sha256(oracle).hexdigest(),**resource))
        for ptr in [records,counts,table]:driver.ok(driver.free(ptr))
        for ptr in [host_records,host_counts]:driver.ok(driver.hostfree(ptr))
    driver.ok(driver.free(data));print('CASE',limit,dynamic,grid,block,'OK',flush=True)
(a.output/f'round-{a.round}.json').write_text(json.dumps(dict(round=a.round,results=rows,checks=checks,passed=True),indent=2)+'\n')
