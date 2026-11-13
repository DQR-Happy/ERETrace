"""Direct Driver API core controls; raw records are validated in RAM, not archived."""
import argparse, ctypes as C, hashlib, json, os, random, re, subprocess, time
from pathlib import Path
import numpy as np
from neutrino.common import Probe, Map, Register
from neutrino.probe.cuda import probing, prune

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--round',type=int,default=1)
p.add_argument('--arch',default='sm_86');p.add_argument('--build',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=True)
# Historical generator only: remove its CLI initialization to import the frozen
# function without producing process.log. Source itself is retained unchanged.
source=a.reference.read_text();source=source.replace("workdir = sys.argv[1]     # directory contains original.bin\nlog = open(os.path.join(workdir, \"process.log\"), 'w')\ntimer = PhaseTimer(workdir, _entry_ns)\ntimer.stage('setup')", "log = sys.stderr")
legacy={'__name__':'frozen_reference'};exec(compile(source,str(a.reference),'exec'),legacy)
maps=[Map('Trace','warp','array',16,1,[Register('start','u64'),Register('elapsed','u32'),Register('sm','u32')])]
probes=[Probe('sched','warp',['kernel'],'mov.u64 %NR0, %clock64;',
    'mov.u64 %NR1, %clock64;\nsub.u64 %NR1, %NR1, %NR0;\n.reg .b32 %core_sm;\nmov.u32 %core_sm, %smid;\ncvt.u64.u32 %NR2, %core_sm;\nSAVE [Trace] {%NR0, %NR1, %NR2};')]
modes=['native','shell','legacy','corrected','shift','late','vector','late_vector']
names=['core_memory','core_pressure','core_reduce']
build=a.output/'build';build.mkdir(exist_ok=True)
if a.build:
    subprocess.run(['g++','-O3','-shared','-fPIC',str(root/'benchmarks/core_repeat.cpp'),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64/stubs','-lcuda','-o',str(build/'repeat.so')],check=True)
    subprocess.run(['nvcc','-O3','-ptx','-arch='+a.arch,str(root/'benchmarks/core_fixture.cu'),'-o',str(build/'fixture.ptx')],check=True)
    text=(build/'fixture.ptx').read_text()
    resources=[]
    for name in names:
        g,f,e,_=prune(text,name)
        for mode in modes:
            if mode=='native':code=e
            elif mode=='shell':code=e.replace(')\n',',\n.param .u64 param_Trace\n)\n',1)
            elif mode=='legacy':code=legacy['probing'](e,probes,maps,3)
            else:code=probing(e,probes,maps,3,mode)
            path=build/f'{name}-{mode}';path.with_suffix('.ptx').write_text(g+f+'\n'+code)
            r=subprocess.run(['ptxas','-arch='+a.arch,'-v',str(path.with_suffix('.ptx')),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True)
            path.with_suffix('.log').write_text(r.stderr);r.check_returncode()
            sass=subprocess.check_output(['cuobjdump','--dump-resource-usage','--dump-sass',str(path.with_suffix('.cubin'))],text=True)
            path.with_suffix('.sass').write_text(sass)
            resources.append(dict(kernel=name,mode=mode,log=r.stderr,static_instructions=len(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@[^ ]+\s+)?[A-Z]',sass)),stores=len(re.findall(r'\bSTG(?:\.|\s)',sass)),sha256=hashlib.sha256(path.with_suffix('.cubin').read_bytes()).hexdigest()))
    (a.output/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')

cu=C.CDLL('libcuda.so.1');ptr=C.c_uint64;handle=C.c_void_p
def bind(n,args):
    f=getattr(cu,n);f.argtypes=args;f.restype=C.c_int;return f
def ok(v):
    if v:raise RuntimeError(f'CUDA error {v}')
init=bind('cuInit',[C.c_uint]);devget=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(handle),C.c_int]);current=bind('cuCtxSetCurrent',[handle]);sync=bind('cuCtxSynchronize',[])
alloc=bind('cuMemAlloc_v2',[C.POINTER(ptr),C.c_size_t]);free=bind('cuMemFree_v2',[ptr]);htod=bind('cuMemcpyHtoD_v2',[ptr,C.c_void_p,C.c_size_t]);dtoh=bind('cuMemcpyDtoH_v2',[C.c_void_p,ptr,C.c_size_t])
load=bind('cuModuleLoad',[C.POINTER(handle),C.c_char_p]);get=bind('cuModuleGetFunction',[C.POINTER(handle),handle,C.c_char_p]);launch=bind('cuLaunchKernel',[handle,*([C.c_uint]*7),handle,C.POINTER(handle),C.POINTER(handle)])
attr=bind('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,handle]);occupancy=bind('cuOccupancyMaxActiveBlocksPerMultiprocessor',[C.POINTER(C.c_int),handle,C.c_int,C.c_size_t])
event=bind('cuEventCreate',[C.POINTER(handle),C.c_uint]);record=bind('cuEventRecord',[handle,handle]);wait=bind('cuEventSynchronize',[handle]);elapsed=bind('cuEventElapsedTime',[C.POINTER(C.c_float),handle,handle]);destroy=bind('cuEventDestroy_v2',[handle])
ok(init(0));dev=C.c_int();ctx=handle();ok(devget(C.byref(dev),0));ok(retain(C.byref(ctx),dev));ok(current(ctx))
functions={};modules=[]
for name in names:
    for mode in modes:
        m=handle();f=handle();ok(load(C.byref(m),str(build/f'{name}-{mode}.cubin').encode()));ok(get(C.byref(f),m,name.encode()));functions[name,mode]=f;modules.append(m)
def call(f,grid,block,args):
    ar=(handle*len(args))(*(C.addressof(x) for x in args));ok(launch(f,*grid,*block,0,None,ar,None))
def mem(n):
    d=ptr();ok(alloc(C.byref(d),n));return d
def read(d,n):
    b=C.create_string_buffer(n);ok(dtoh(b,d,n));return b.raw
def upload(d,b):ok(htod(d,b,len(b)))

repeater=C.CDLL(str(build/'repeat.so')).core_repeat
repeater.argtypes=[handle,C.c_int,C.c_int,C.c_int,ptr,ptr,C.c_int,C.c_int,C.POINTER(C.c_float)];repeater.restype=C.c_int
# Warm the device before any measured condition; no fixed clocks are assumed.
warm=mem(82*1024*4);upload(warm,bytes(82*1024*4));warm_ms=C.c_float();ok(repeater(functions['core_pressure','native'],82,1024,64,warm,warm,0,10000,C.byref(warm_ms)));ok(free(warm))
rows=[];checks=[];rng=random.Random(21042+a.round)
# Explicit partial-warp and 3D index oracle: payload fields are identities, not
# nondeterministic clocks. Legacy overlaps are recorded as expected failures.
identity=Probe('identity','warp',['kernel'],None,
    'mov.u32 %Trace_u32_1, %ctaid.x;\ncvt.u64.u32 %NR0, %Trace_u32_1;\nmov.u32 %Trace_u32_1, %tid.x;\ncvt.u64.u32 %NR1, %Trace_u32_1;\nmov.u64 %NR2, 123;\nSAVE [Trace] {%NR0, %NR1, %NR2};')
simple='.version 8.4\n.target '+a.arch+'\n.address_size 64\n.visible .entry ids(\n.param .u64 dummy\n)\n{\nret;\n}'
for mode in ['legacy','corrected','shift','vector']:
    code=legacy['probing'](simple,[identity],maps,3) if mode=='legacy' else probing(simple,[identity],maps,3,mode)
    pt=build/f'ids-{mode}.ptx';pt.write_text(code);subprocess.run(['ptxas','-arch='+a.arch,str(pt),'-o',str(pt.with_suffix('.cubin'))],check=True)
    m=handle();f=handle();ok(load(C.byref(m),str(pt.with_suffix('.cubin')).encode()));ok(get(C.byref(f),m,b'ids'));modules.append(m)
    for threads in [1,31,32,33,48,63,64,65,96,127,128,256,1024]:
        count=3*((threads+31)//32);d=mem(count*16+64);upload(d,b'\xa5'*(count*16+64));call(f,(3,1,1),(threads,1,1),[d,d]);raw=read(d,count*16+64)
        records=np.frombuffer(raw[:count*16],dtype=[('cta','<u8'),('tid','<u4'),('tag','<u4')])
        expected=[(b,t,123) for b in range(3) for t in range(0,threads,32)]
        correct=records.tolist()==expected;guard=raw[-64:]==b'\xa5'*64
        assert guard and (correct if mode!='legacy' or threads%32==0 else not correct)
        checks.append(dict(mode=mode,threads=threads,records=count,exact=correct,guard_ok=guard));ok(free(d))

# Kernel-time control: repeated launches intentionally reuse state, equally for
# all variants; each timing group starts from the identical reset input. Results
# are compared after a single launch separately (no count replay).
shapes=[(82,128,1),(328,256,16),(82,1024,64)]
for name in names:
    for grid,block,loops in shapes:
        n=grid*block;size=grid*((block+31)//32)*16;data=mem(n*4);trace=mem(size+64);initial=np.arange(n,dtype=np.uint32).tobytes();oracle=None
        upload(data,initial);call(functions[name,'native'],(grid,1,1),(block,1,1),[data,C.c_int(loops)]);ok(sync());oracle=read(data,n*4)
        order=modes[:];rng.shuffle(order)
        for mode in order:
            f=functions[name,mode];args=[data,C.c_int(loops)]+([trace] if mode!='native' else [])
            upload(data,initial);upload(trace,b'\xa5'*(size+64));call(f,(grid,1,1),(block,1,1),args);ok(sync());result=read(data,n*4)
            if oracle is None:oracle=result
            assert result==oracle,(name,mode,'model output')
            tr=read(trace,size+64);assert tr[-64:]==b'\xa5'*64
            if mode not in ['native','shell']:
                rr=np.frombuffer(tr[:size],dtype=[('start','<u8'),('elapsed','<u4'),('sm','<u4')]);assert np.all(rr['start']!=0) and np.all(rr['sm']<256) and np.all(rr['elapsed']>0)
            regs=C.c_int();local=C.c_int();occ=C.c_int();ok(attr(C.byref(regs),4,f));ok(attr(C.byref(local),3,f));ok(occupancy(C.byref(occ),f,block,0))
            samples=[]
            for repeat in range(5):
                upload(data,initial)
                for _ in range(10):call(f,(grid,1,1),(block,1,1),args)
                ok(sync());ms=C.c_float();ok(repeater(f,grid,block,loops,data,trace,int(mode!='native'),500,C.byref(ms)));samples.append(ms.value/500)
            rows.append(dict(kernel=name,grid=grid,block=block,loops=loops,mode=mode,samples_ms=samples,median_ms=float(np.median(samples)),registers=regs.value,local_bytes=local.value,max_blocks_per_sm=occ.value,trace_bytes=size,output_digest=hashlib.sha256(result).hexdigest()))
        ok(free(data));ok(free(trace))
(a.output/f'round-{a.round}.json').write_text(json.dumps(dict(round=a.round,checks=checks,results=rows),indent=2)+'\n')
print(json.dumps(dict(round=a.round,checks=len(checks),timed=len(rows))),flush=True)
