import argparse,ctypes as C,json,math,os,struct,time,hashlib
from pathlib import Path
import numpy as np
from core_driver import Driver
p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--fatbin',required=True);p.add_argument('--xtrace');p.add_argument('--out',required=True);p.add_argument('--trace-dir',required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--gpu-profile');p.add_argument('--event-profile',action='store_true');p.add_argument('--native-decode',action='store_true');p.add_argument('--detail',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);d=Driver()
hook=C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER']) if a.mode in ('U','R') else None
blob=C.create_string_buffer(Path(a.fatbin).read_bytes());module=C.c_void_p();f=C.c_void_p();load=d.lib.cuModuleLoadData;load.argtypes=[C.POINTER(C.c_void_p),C.c_void_p];load.restype=C.c_int;d.ok(load(C.byref(module),blob));d.ok(d.get(C.byref(f),module,b'scoped_rmsnorm'))
x=np.linspace(-1,1,1024,dtype=np.float32);inp=d.memory(4096);dest=d.memory(4096);d.upload(inp,x.tobytes());trace=None
if a.mode=='X':f=d.function(a.xtrace,'scoped_rmsnorm');schema=json.loads(Path(a.xtrace).with_suffix('.json').read_text());trace=d.memory(schema['buffer_bytes']);d.upload(trace,bytes(schema['buffer_bytes']));sink=C.CDLL(str(Path(a.xtrace).with_name('scope_sink.so')));sink.scope_sink.argtypes=[C.c_void_p,C.c_size_t,C.c_char_p];sink.scope_sink.restype=C.c_int
if hook:hook.neutrino_ablation_flush.argtypes=[C.c_char_p]
profile=None
if a.gpu_profile:
 profile=C.CDLL(a.gpu_profile);profile.deep_stop.argtypes=[C.c_char_p];assert profile.deep_init()==0
event_ns=[]
if a.event_profile:
 create=d.lib.cuEventCreate;create.argtypes=[C.POINTER(C.c_void_p),C.c_uint];create.restype=C.c_int
 record=d.lib.cuEventRecord;record.argtypes=[C.c_void_p,C.c_void_p];record.restype=C.c_int
 elapsed=d.lib.cuEventElapsedTime;elapsed.argtypes=[C.POINTER(C.c_float),C.c_void_p,C.c_void_p];elapsed.restype=C.c_int
 ev0=C.c_void_p();ev1=C.c_void_p();d.ok(create(C.byref(ev0),0));d.ok(create(C.byref(ev1),0))
trace_dir=Path(a.trace_dir);os.environ['SCOPE_TRACE_DIR']=str(trace_dir)
decoder=None
if a.native_decode:
 from gwatch.cuda.xtrace._ptx_instrumenter import TraceInstrumenter_PTX
 from gwatch.cuda.xtrace._schema import TraceSchema_PTX
 from gwatch.cuda.xtrace._ptx_sites import schedule_sites
 import gwatch.cuda.binary as binary
 b=binary.PTX();b.fill(str(Path(a.fatbin).with_suffix('.ptx')));b.parse();k=b.get_kerneldef_by_name('scoped_rmsnorm');sites=schedule_sites(k)
 for site in sites:site.value_base='%smid';site.value_base_bits=32;site.value_width_bits=32
 counts={(x.type_str,x.scope_str):1 for x in sites}
 native_schema=TraceSchema_PTX({x.pc:x for x in sites},(1,1,1),(1024,1,1),per_thread_counts=counts,warp_level=True)
 class Buffer:
  def to_numpy(self):return np.frombuffer(d.read(trace,schema['buffer_bytes']),dtype=np.uint8)
 decoder=TraceInstrumenter_PTX('scoped_rmsnorm',k,b,Buffer(),native_schema,'sm',sites)
phase_samples=[]

def step(i):
 begin=time.perf_counter_ns()
 if a.event_profile:d.ok(record(ev0,None))
 d.call(f,(1,1,1),(1024,1,1),[inp,dest]+([trace] if trace else []))
 if a.event_profile:d.ok(record(ev1,None))
 submit_end=time.perf_counter_ns() if a.detail else 0
 d.ok(d.sync())
 sync_end=time.perf_counter_ns() if a.detail else 0
 if hook:assert hook.neutrino_ablation_boundary()==0
 boundary_end=time.perf_counter_ns() if a.detail else 0
 if trace and not decoder:
  data=d.read(trace,schema['buffer_bytes']);assert sink.scope_sink(C.c_char_p(data),schema['stride_bytes'],str(trace_dir/f'xtrace-{i}.raw').encode())==0

 if decoder:
  records=decoder.collect();assert len(records)==64
  entry={r['global_tid']:r for r in records if r['type_str']=='SCOPE_START'}
  exit_={r['global_tid']:r for r in records if r['type_str']=='SCOPE_END'}
  assert set(entry)==set(exit_)==set(range(0,1024,32))
  payload=b''.join(struct.pack('<QII',entry[t]['payload'],exit_[t]['payload']-entry[t]['payload'],exit_[t]['value']) for t in range(0,1024,32))
  with (trace_dir/f'xtrace-{i}.raw').open('wb') as fp:assert fp.write(payload)==512
 end=time.perf_counter_ns();duration=end-begin
 if a.detail:phase_samples.append(dict(submit_ns=submit_end-begin,app_sync_ns=sync_end-submit_end,drain_ns=boundary_end-sync_end,external_copy_decode_file_ns=end-boundary_end,total_ns=duration))
 if a.event_profile:
  ms=C.c_float();d.ok(elapsed(C.byref(ms),ev0,ev1));event_ns.append(ms.value*1e6)
 return duration
cold=step(-1)
for i in range(10):step(-2-i)
if hook:
 assert hook.neutrino_ablation_reset()==0
 if os.environ.get('NEUTRINO_TIMING')=='1':assert hook.neutrino_timing_reset()==0
if profile:assert profile.deep_start()==0
event_ns.clear();phase_samples.clear()
samples=[step(i) for i in range(a.steps)]
if profile:assert profile.deep_stop(str(out/'gpu.json').encode())==0
actual=np.frombuffer(d.read(dest,4096),dtype=np.float32);expected=x/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(1e-5));assert np.allclose(actual,expected,rtol=1e-5,atol=1e-6)
if hook:
 if os.environ.get('NEUTRINO_ABLATION_STATS')=='1':assert hook.neutrino_ablation_flush(str(out/'capture.json').encode())==0
 if os.environ.get('NEUTRINO_TIMING')=='1':assert hook.neutrino_timing_flush()==0
 if os.environ.get('NEUTRINO_TINY_SCOPE')=='1':assert hook.neutrino_tiny_scope_release()==0
if a.event_profile:
 destroy=d.lib.cuEventDestroy_v2;destroy.argtypes=[C.c_void_p];destroy.restype=C.c_int;d.ok(destroy(ev0));d.ok(destroy(ev1))
 (out/'event.json').write_text(json.dumps(dict(samples_ns=event_ns,mean_us=sum(event_ns)/len(event_ns)/1000,kernels=a.steps,scope='CUDA Event envelope surrounding the host launch call; includes submission gaps and hook callback work, not pure kernel Activity'))+'\n')
if a.detail:(out/'host-phases.json').write_text(json.dumps(dict(samples=phase_samples,mean_ns={k:sum(x[k] for x in phase_samples)/len(phase_samples) for k in phase_samples[0]}),indent=2)+'\n')
(out/'workload.json').write_text(json.dumps(dict(mode=a.mode,steps=a.steps,cold_first_step_ns=cold,warmup_steps=11,samples_ns=samples,mean_us=sum(samples)/len(samples)/1000,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),numeric_correct=True,logical_records_per_launch=32,logical_trace_bytes_per_launch=512,clock='clock64',fields=['start:u64','elapsed:u32','sm:u32'],xtrace_schema=schema if trace else None),indent=2)+'\n')
