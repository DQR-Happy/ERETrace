"""Real-PTX certified event layouts + actual-count compaction, isolated instances."""
import argparse,ctypes as C,hashlib,json,math,os,random,subprocess,time
from pathlib import Path
import numpy as np
from core_driver import Driver,P,H
from table4_cases import cases,scalar_params,arguments,event_counts
from neutrino.probe.core_bounds import specialize,Program
from neutrino.probe.core_certified import compile_layout
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=7);p.add_argument('--build',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--samples',type=int,default=7);p.add_argument('--only',nargs='*');p.add_argument('--round-index',type=int,default=1);p.add_argument('--build-dir',type=Path);p.add_argument('--modes',nargs='+',choices=['T','C'],default=['T','C']);a=p.parse_args()
assert set(a.modes)=={'T','C'} and len(a.modes)==2
a.output.mkdir(parents=True,exist_ok=True);build=a.build_dir or a.output/'build';build.mkdir(exist_ok=True);root=Path(__file__).resolve().parents[1]
cs=[c for c in cases(a.inputs) if not a.only or c['tag'] in a.only]
if a.smoke:cs=[cs[0],cs[-1]]
compilation=[]
if a.build:
 subprocess.run(['nvcc','-O3','-std=c++17','-arch=sm_86','-shared','-Xcompiler','-fPIC',str(root/'benchmarks/certified_compact.cu'),'-lcuda','-o',str(build/'compact.so')],check=True)
 for case in cs:
  t=time.perf_counter_ns();params=scalar_params(case);cert=specialize(case['ptx'],params,(case['grid'],1,1),(case['block'],1,1));planning=time.perf_counter_ns()-t
  (build/(case['tag']+'-certificate.json')).write_text(json.dumps(cert,indent=2)+'\n')
  item=dict(tag=case['tag'],planning_ns=planning,capacities_min=min(cert['capacities']),capacities_max=max(cert['capacities']),slots=cert['slots'],control_classes=cert['control_classes'],abstract_states=cert['abstract_states'],source_sha256=cert['ptx_sha256'],variants={})
  for mode in ['N','U','T','C']:
   if mode=='N':code=case['ptx'];layout={}
   else:code,layout=compile_layout(case['ptx'],cert,params,(case['grid'],1,1),(case['block'],1,1),dict(U='uniform',T='table',C='classes')[mode])
   tag=case['tag']+'-'+mode;path=build/(tag+'.ptx');path.write_text(code)
   r=subprocess.run(['ptxas','-arch=sm_86','-v',str(path),'-o',str(path.with_suffix('.cubin'))],text=True,capture_output=True);path.with_suffix('.log').write_text(r.stderr);r.check_returncode()
   if layout:(build/(tag+'-layout.json')).write_text(json.dumps(layout)+'\n')
   item['variants'][mode]=dict(ptx_sha256=hashlib.sha256(code.encode()).hexdigest(),cubin_sha256=hashlib.sha256(path.with_suffix('.cubin').read_bytes()).hexdigest())
  compilation.append(item);print('BUILT',case['tag'],item['capacities_min'],item['capacities_max'],round(planning/1e9,3),flush=True)
 (a.output/'compilation.json').write_text(json.dumps(compilation,indent=2)+'\n')

driver=Driver();lib=C.CDLL(str(build/'compact.so'))
class Result(C.Structure):_fields_=[(x,C.c_uint64) for x in ['total_ns','kernel_wait_ns','compact_ns','metadata_ns','copy_ns','bytes','events']]+[(x,C.c_float) for x in ['kernel_ms','scan_ms','gather_ms']]+[('overflow',C.c_uint64)]
lib.cert_create.argtypes=[C.c_uint,C.c_size_t,C.c_int,H,H,C.POINTER(H)];lib.cert_create.restype=C.c_int
lib.cert_bind.argtypes=[H,H,C.c_uint,C.c_uint,H,C.c_uint];lib.cert_bind.restype=C.c_int
lib.cert_run.argtypes=[H,H,C.c_uint,C.c_uint,C.POINTER(H),P,P,C.c_uint,C.c_int,C.c_int,C.POINTER(Result)];lib.cert_run.restype=C.c_int
for fn in ['cert_host','cert_counts']:getattr(lib,fn).argtypes=[H];getattr(lib,fn).restype=H
lib.cert_space.argtypes=[H];lib.cert_space.restype=C.c_size_t;lib.cert_destroy.argtypes=[H]
lib.cert_workspace.argtypes=[H];lib.cert_workspace.restype=C.c_size_t
lib.cert_gpu_repeat.argtypes=[H,C.c_uint,C.c_uint,C.c_uint,C.POINTER(H),C.c_uint,C.POINTER(C.c_float)];lib.cert_gpu_repeat.restype=C.c_int
dtype=np.dtype([('time','<u8'),('address','<u8'),('site','<u4'),('thread','<u4'),('mask','<u4'),('width','<u4')])

def launch(fn,case,args):
 arr=(H*len(args))(*(C.addressof(x) for x in args));driver.ok(driver.launch(fn,case['grid'],1,1,case['block'],1,1,case['shared'],None,arr,None));driver.ok(driver.sync())
def oracle_records(case,raw,counts,offsets,caps,ptrs,sites,compact):
 n=case['n'];threads=case['grid']*case['block'];ids=np.arange(threads);local=ids%case['block'];row=ids//case['block']
 if case['kind']=='rms':expected=4*np.maximum(0,(n-1-local)//case['block']+1)
 elif case['kind']=='silu':expected=np.where(ids<n,3,0)
 else:expected=event_counts(case)
 assert np.array_equal(counts,expected),(case['tag'],'counts',counts[:5],expected[:5]);assert np.all(counts<=caps)
 if compact:offsets=np.r_[0,np.cumsum(counts[:-1])]
 rec=np.frombuffer(raw,dtype=dtype);retained=int(expected.sum());assert np.all(rec['time'][np.concatenate([np.arange(int(o),int(o+c)) for o,c in zip(offsets,counts)])]>0)
 op={s['site']:s['opcode'].split('.')[0] for s in sites};checked=0
 if case['kind'] in ['density','irregular']:
  for lo in range(0,threads,8192):
   hi=min(threads,lo+8192);cs=counts[lo:hi].astype('int64');own=np.repeat(np.arange(lo,hi),cs)
   starts=np.cumsum(cs)-cs;js=np.arange(int(cs.sum()))-np.repeat(starts,cs)
   ix=np.repeat(offsets[lo:hi],cs)+js;rr=rec[ix.astype('int64')]
   expected_sites=np.where(js==0,0,np.where(js%2==1,1,2)).astype('uint32')
   expected_addr=ptrs[0].value+own*8+np.where(js==0,0,4)
   assert np.all(rr['time']>0) and np.array_equal(rr['thread'],own)
   assert np.array_equal(rr['site'],expected_sites) and np.array_equal(rr['address'],expected_addr)
   assert np.all(rr['width']==4) and np.all((rr['mask']>>((own%case['block'])%32))&1)
   checked+=len(rr)
  return checked
 for i,(o,c) in enumerate(zip(offsets,counts)):
  rr=rec[int(o):int(o+c)];assert np.all(rr['thread']==i) and np.all(rr['width']==4)
  assert np.all((rr['mask']>>int(local[i]%32))&1)
  if case['kind']=='rms':
   cols=list(range(int(local[i]),n,case['block']));expected_a=[ptrs[0].value+(int(row[i])*n+x)*4 for x in cols]
   for x in cols:expected_a.extend([ptrs[0].value+(int(row[i])*n+x)*4,ptrs[1].value+x*4,ptrs[2].value+(int(row[i])*n+x)*4])
   expected_op=['ld']*len(cols)+['ld','ld','st']*len(cols)
  elif case['kind']=='silu':expected_a=[ptrs[j].value+i*4 for j in range(3)] if i<n else [];expected_op=['ld','ld','st'] if i<n else []
  else:expected_a=[ptrs[0].value+i*8]+[ptrs[0].value+i*8+4]*(int(c)-1);expected_op=['ld']+['ld','st']*case['density']+['ld','st']
  assert rr['address'].tolist()==expected_a,(case['tag'],i,'address');assert [op[int(s)] for s in rr['site']]==expected_op
  checked+=int(c)
 return checked

allrows=[];checks=[]
for ri in range(a.round_index,a.round_index+a.rounds):
 rng=random.Random(21042+ri)
 for case in cs:
  names=Program(case['ptx']);n=case['n'];threads=case['grid']*case['block']
  rngnp=np.random.default_rng(21042)
  if case['kind']=='rms':blobs=[rngnp.normal(size=n*case['grid']).astype('f4').tobytes(),rngnp.uniform(.5,1.5,size=1<<(n-1).bit_length()).astype('f4').tobytes(),bytes(n*case['grid']*4)];output=2
  elif case['kind']=='silu':blobs=[rngnp.normal(size=n).astype('f4').tobytes(),rngnp.normal(size=n).astype('f4').tobytes(),bytes(n*4)];output=2
  else:
   data=np.zeros((threads,2),dtype='u4');data[:,0]=case['density'];data[:,1]=np.arange(threads);blobs=[data.tobytes()];output=0
  ptrs=[driver.memory(len(b)) for b in blobs]
  for ptr,b in zip(ptrs,blobs):driver.upload(ptr,b)
  args=arguments(case,ptrs);native=driver.function(build/(case['tag']+'-N.cubin'),names.name)
  launch(native,case,args);expected_output=driver.read(ptrs[output],len(blobs[output]));digest=hashlib.sha256(expected_output).hexdigest()
  if case['kind']=='rms':
   x=np.frombuffer(blobs[0],dtype='f4').reshape(-1,n);w=np.frombuffer(blobs[1],dtype='f4')[:n];ref=x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-5)*w
   np.testing.assert_allclose(np.frombuffer(expected_output,dtype='f4').reshape(-1,n),ref,rtol=3e-5,atol=3e-5)
  elif case['kind']=='silu':
   x=np.frombuffer(blobs[0],dtype='f4');y=np.frombuffer(blobs[1],dtype='f4');np.testing.assert_allclose(np.frombuffer(expected_output,dtype='f4'),x/(1+np.exp(-x))*y,rtol=3e-5,atol=3e-5)
  modes=list(a.modes);rng.shuffle(modes) if ri==0 or ri==99 else None; shift=(ri-1)%len(modes); modes=modes[shift:]+modes[:shift]
  for mode in modes:
   actual='C' if mode=='K' else mode;lay=json.loads((build/(case['tag']+'-'+actual+'-layout.json')).read_text()) if mode!='N' else dict(slots=1,caps=[0],offsets=[0],plan=[])
   slots=lay['slots'];records=driver.memory(slots*32+128);counts=driver.memory(threads*8+128);plan_raw=np.array(lay['plan'],dtype='<u8').tobytes();plan=driver.memory(max(1,len(plan_raw)))
   if plan_raw:driver.upload(plan,plan_raw)
   fn=driver.function(build/(case['tag']+'-'+actual+'.cubin'),names.name)
   pargs=args+([] if mode=='N' else [records]+([plan] if actual in ['U','T','C'] else [])+[counts]);arr=(H*len(pargs))(*(C.addressof(x) for x in pargs))
   engine=H();off=np.asarray(lay['offsets'],dtype='u8');cap=np.asarray(lay['caps'],dtype='u8')
   t=time.perf_counter_ns();driver.ok(lib.cert_create(1 if mode=='N' else threads,slots,int(mode=='K'),off.ctypes.data,cap.ctypes.data,C.byref(engine)));setup=time.perf_counter_ns()-t
   if mode!='N':
    bindings=[]
    for key,value in scalar_params(case).items():
     nm,sep,ofs=key.partition('+');idx=names.params.index(nm);offset=int(ofs) if sep else 0
     size=4 if isinstance(args[idx],C.Structure) else C.sizeof(args[idx])
     bits=value&((1<<(size*8))-1);bindings.append((idx,offset,size,bits))
    bind_raw=np.asarray(bindings,dtype='u8');driver.ok(lib.cert_bind(engine,fn,case['grid'],case['block'],bind_raw.ctypes.data,len(bindings)))
   def once(diag=0):
    if case['kind'] in ['density','irregular']:driver.upload(ptrs[0],blobs[0])
    res=Result();driver.ok(lib.cert_run(engine,fn,case['grid'],case['block'],arr,records,counts,case['shared'],0 if mode=='N' else 2 if mode=='K' else 1,diag,C.byref(res)));return {k:getattr(res,k) for k,_ in Result._fields_}
   driver.upload(records,b'\xa5'*(slots*32+128));driver.upload(counts,b'\xa5'*(threads*8+128));check=once()
   assert driver.read(ptrs[output],len(blobs[output]))==expected_output
   if mode!='N':
    cnt=np.frombuffer(C.string_at(lib.cert_counts(engine),threads*8),dtype='<u8');size=check['events']*32 if mode=='K' else slots*32
    raw=C.string_at(lib.cert_host(engine),size);validated=oracle_records(case,raw,cnt,off,cap,ptrs,names.sites,mode=='K')
    assert not check['overflow'];assert driver.read(P(records.value+slots*32),128)==b'\xa5'*128 and driver.read(P(counts.value+threads*8),128)==b'\xa5'*128
    if mode=='K':
     original=driver.read(records,slots*32);rr=np.frombuffer(original,dtype=dtype);joined=b''.join(rr[int(o):int(o+c)].tobytes() for o,c in zip(off,cnt));assert joined==raw
    checks.append(dict(round=ri,case=case['tag'],mode=mode,records=validated,complete=True,output_digest=digest,same_map_compaction=mode=='K'))
   gpu_samples=[]
   for _ in range(3):
    if case['kind'] in ['density','irregular']:driver.upload(ptrs[0],blobs[0])
    us=C.c_float();driver.ok(lib.cert_gpu_repeat(fn,case['grid'],case['block'],case['shared'],arr,16,C.byref(us)));gpu_samples.append(us.value)
   once();once();samples=[once() for _ in range(a.samples)];diag=[once(1) for _ in range(3)]
   row=dict(round=ri,case=case['tag'],mode=mode,gpu_kernel_samples_us=gpu_samples,gpu_kernel_mean_us=float(np.mean(gpu_samples)),workspace_bytes=lib.cert_workspace(engine),threads=threads,grid=case['grid'],block=case['block'],capacity_min=min(lay['caps']),capacity_max=max(lay['caps']),actual_events=int(event_counts(case).sum()),reserved_slots=slots,samples=samples,diagnostic=diag,mean_ns=float(np.mean([s['total_ns'] for s in samples])),median_ns=float(np.median([s['total_ns'] for s in samples])),record_bytes=slots*32 if mode!='N' else 0,count_bytes=threads*8 if mode!='N' else 0,plan_bytes=len(plan_raw),compaction_extra_device_bytes=lib.cert_space(engine),host_receive_bytes=slots*32+(1 if mode=='N' else threads)*8+144,setup_ns=setup,output_digest=digest,**driver.resources(fn,case['block']))
   allrows.append(row);print('ROUND',ri,case['tag'],mode,round(row['mean_ns']/1e3,2),'us',flush=True)
   lib.cert_destroy(engine)
   for ptr in [records,counts,plan]:driver.ok(driver.free(ptr))
  for ptr in ptrs:driver.ok(driver.free(ptr))
 (a.output/'results.json').write_text(json.dumps(dict(passed=True,rows=allrows,checks=checks,rounds_completed=ri),indent=2)+'\n')
(a.output/'complete.json').write_text(json.dumps(dict(passed=True,rounds=a.rounds,cases=len(cs),measurements=len(allrows),raw_trace_saved=False))+'\n')
