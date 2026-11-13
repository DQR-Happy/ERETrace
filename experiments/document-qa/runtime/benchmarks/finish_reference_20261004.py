"""Serial independent confirmation and mechanistic controls after screening."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable
assert (out/'reference-screen/complete.json').exists()
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def call(cmd,log):
 print('START',log,flush=True)
 with (root/log).open('w') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=f,check=True)
 print('DONE',log,flush=True)
def matrix(name,conditions,rounds,*args):
 if (out/name/'complete.json').exists():return
 call([py,'benchmarks/run_reference_20261004.py','--name',name,'--conditions',*conditions.split(),'--rounds',str(rounds),*args],name+'.log')
matrix('confirmation-7round','N HOLD H2 R0 T4 BP T4P',7)
matrix('context800-5round','N HOLD BP T4P',5,'--repeat','128')
matrix('model17-5round','N HOLD BP T4P',5,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-1.7B-Q4_K_M.gguf')
matrix('llama1b-5round','N HOLD BP T4P',5,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf')
matrix('sustained-3round','N HOLD BP T4P',3,'--requests','20')
matrix('host-detail-3round','H2 R0 T4 T8 BP T4P',3,'--diagnostic','--requests','1','--layout')
matrix('gpu-detail-3round','N H2 R0 T4 BP T4P',3,'--gpu','--requests','1')
matrix('observer-5round','BP BPD',5)
(out/'performance-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
call(['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-gencode','arch=compute_86,code=[sm_86,compute_86]','benchmarks/reference_shape_workload.cu','-ldl','-o','benchmarks/reference_shape_workload'],'shape-build.log')
for name in ['shapes','tail','stateful']:
 if not (out/f'reference-{name}/summary.json').exists():call([py,f'benchmarks/run_reference_{name}.py'],f'reference-{name}.log')
for test in ['test_timing.py','test_arena_ring.py','test_capture_codec.py','test_gpu9_format.py','test_hier_format.py','test_reference_format.py']:
 call([py,'-m','unittest','discover','-s','tests','-p',test],test+'.log')
if not (out/'default-gpu-regression/summary.json').exists():call([py,'tests/run_gpu_validation.py','--output',str(out/'default-gpu-regression'),'--trace-root','/dev/shm/neutrino-reference-20261004'],'default-regression.log')
call(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'benchmarks/gpu_reference_smoke.py','benchmarks/capture_pack.cubin',str(out/'reference-memcheck.json')],'reference-memcheck.log')
call([py,'benchmarks/verification_provenance.py'],'provenance.log')
raw=Path('/dev/shm/neutrino-reference-20261004');assert not any(raw.iterdir())
(out/'cleanup.json').write_text(json.dumps(dict(raw_trace_remaining=0,root=str(raw)))+'\n')
(out/'suite-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
