"""Held-out writer service tests, scheduled after primary matrices/microbenchmarks."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable;start=time.monotonic();absent=0
while not (out/'suite-complete.json').exists():
    running=subprocess.run(['pgrep','-f','[f]inish_verification_20261004.py'],stdout=subprocess.DEVNULL).returncode==0
    absent=0 if running else absent+1
    if absent>6 or time.monotonic()-start>5400:raise RuntimeError('Main finish stage stopped before completion')
    time.sleep(5)
for name,extra in [('writer-heldout-context',['--repeat','128']),('writer-heldout-model17',['--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-1.7B-Q4_K_M.gguf'])]:
    subprocess.run([py,'benchmarks/run_verification_20261004.py','--name',name,'--conditions','DW','--rounds','3','--requests','1','--diagnostic',*extra],cwd=root,check=True)
subprocess.run([py,'benchmarks/verification_provenance.py'],cwd=root,check=True)
assert not any(Path('/dev/shm/neutrino-verify-20261004').iterdir())
(out/'writer-model-complete.json').write_text(json.dumps(dict(complete=True,raw_trace_remaining=0))+'\n')
