"""Serial follow-up; never profiles concurrently with main performance windows."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable
start=time.monotonic();absent=0
while not (out/'model17-5round/complete.json').exists():
 running=subprocess.run(['pgrep','-f','[r]un_mechanism_20261004.py'],stdout=subprocess.DEVNULL).returncode==0
 absent=0 if running else absent+1
 if absent>10 or time.monotonic()-start>7200:raise RuntimeError('Main matrix stopped or timed out')
 time.sleep(3)
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def call(cmd,log):
 print('START',log,flush=True)
 with (root/log).open('w') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=f,check=True)
 print('DONE',log,flush=True)
def matrix(name,conditions,rounds,*args):
 if (out/name/'complete.json').exists():return
 call([py,'benchmarks/run_mechanism_20261004.py','--name',name,'--conditions',*conditions.split(),'--rounds',str(rounds),*args],name+'.log')
# The earlier context800 directory actually used repeat=110; retain it, and
# independently run the verified repeat=128 that gives 800 tokenizer tokens.
matrix('context128-5round','N DW F32 H3 H2',5,'--repeat','128')
matrix('llama1b-5round','N DW H3 H2',5,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf')
matrix('sustained-3round','N DW H2',3,'--requests','20')
matrix('sync-anchor-7round','N S DW H2',7)
matrix('host-detail-3round','DW F32 H2 H3 F23 DWM H2R',3,'--diagnostic','--requests','1','--layout')
matrix('gpu-detail-3round','N DW F32 H2 H3',3,'--gpu','--requests','1')
matrix('observer-5round','H2 H2D',5)
call([py,'benchmarks/run_profiling_tail.py'],'profiling-tail.log')
call([py,'benchmarks/run_mechanism_stateful.py'],'mechanism-stateful.log')
for test in ['test_timing.py','test_arena_ring.py','test_capture_codec.py','test_gpu9_format.py','test_hier_format.py']:
 call([py,'-m','unittest','discover','-s','tests','-p',test],test+'.log')
call([py,'tests/run_gpu_validation.py','--output',str(out/'default-gpu-regression'),'--trace-root','/dev/shm/neutrino-mechanism-20261004'],'default-regression.log')
for name in ['flex','hier']:
 call(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,f'benchmarks/gpu_{name}_smoke.py','benchmarks/capture_pack.cubin',str(out/f'{name}-memcheck.json')],f'{name}-memcheck.log')
call([py,'benchmarks/verification_provenance.py'],'provenance.log')
raw=Path('/dev/shm/neutrino-mechanism-20261004');assert not any(raw.iterdir())
(out/'cleanup.json').write_text(json.dumps(dict(raw_trace_remaining=0,root=str(raw)))+'\n')
(out/'suite-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
