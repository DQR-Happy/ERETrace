"""Holdout workloads and correctness after choosing the 16KiB knee."""
import json,os,statistics,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable
deadline=time.monotonic()+7200;absent=0
while not (out/'confirmation-complete.json').exists():
 alive=subprocess.run(['pgrep','-f','[f]inish_reference_cutoff_confirm.py'],stdout=subprocess.DEVNULL).returncode==0
 absent=0 if alive else absent+1
 if absent>6 or time.monotonic()>deadline:raise RuntimeError('Confirmation stopped')
 time.sleep(5)
rs=json.loads((out/'cutoff-confirmation-7round/runs.json').read_text());means={c:statistics.mean(r['wall_ns']/len(r['requests']) for r in rs if r['condition']==c) for c in ['K32','K16','K8','K4']}
assert min(means,key=means.get)=='K16'
(out/'selection.json').write_text(json.dumps(dict(selected='K16',means_ns=means,reason='Lowest independent-confirmation mean; differences within 4/8/16KiB are not established as unique optima. 16KiB uses fewer coded Maps.'),indent=2)+'\n')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def call(cmd,log):
 print('START',log,flush=True)
 with (root/log).open('w') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=f,check=True)
 print('DONE',log,flush=True)
def matrix(name,conditions,rounds,*args):
 if (out/name/'complete.json').exists():return
 call([py,'benchmarks/run_reference_20261004.py','--name',name,'--conditions',*conditions.split(),'--rounds',str(rounds),*args],name+'.log')
matrix('final-anchor-7round','N HOLD BOLD K16',7)
matrix('cutoff-context800-5round','N BOLD K16',5,'--repeat','128')
matrix('cutoff-model17-5round','N BOLD K16',5,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-1.7B-Q4_K_M.gguf')
matrix('cutoff-llama1b-5round','N BOLD K16',5,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf')
matrix('cutoff-sustained-3round','N BOLD K16',3,'--requests','20')
matrix('cutoff-observer-5round','K16 K16D',5)
matrix('cutoff-context-validation','K16',1,'--repeat','128','--requests','1','--validate')
matrix('cutoff-model17-validation','K16',1,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-1.7B-Q4_K_M.gguf','--requests','1','--validate')
for test in ['test_timing.py','test_arena_ring.py','test_capture_codec.py','test_gpu9_format.py','test_hier_format.py','test_reference_format.py']:
 call([py,'-m','unittest','discover','-s','tests','-p',test],test+'.log')
if not (out/'default-gpu-regression/summary.json').exists():call([py,'tests/run_gpu_validation.py','--output',str(out/'default-gpu-regression'),'--trace-root','/dev/shm/neutrino-reference-20261004'],'default-regression.log')
# The helper binary is unchanged from V1; keep a second memcheck completion
# record for this actual build rather than implying the whole pipeline is checked.
call(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'benchmarks/gpu_reference_smoke.py','benchmarks/capture_pack.cubin',str(out/'reference-memcheck.json')],'reference-memcheck.log')
call([py,'benchmarks/verification_provenance.py'],'provenance.log')
raw=Path('/dev/shm/neutrino-reference-20261004');assert not any(raw.iterdir())
(out/'cleanup.json').write_text(json.dumps(dict(raw_trace_remaining=0,root=str(raw)))+'\n')
(out/'suite-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
