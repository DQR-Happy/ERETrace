import json,os,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/neutrino-verify-codec-20261004');start=time.monotonic()
while not (root/'results/observer-5round/complete.json').exists():
    if time.monotonic()-start>1800:raise RuntimeError('Observer comparison incomplete')
    time.sleep(5)
py='/root/autodl-tmp/neutrino-repro/.venv/bin/python'
subprocess.run([py,'benchmarks/run_verification_20261004.py','--name','template-anchor-7round','--conditions','N','I','P','DW','--rounds','7'],cwd=root,check=True)
subprocess.run([py,'benchmarks/verification_provenance.py'],cwd=root,check=True)
assert not any(Path('/dev/shm/neutrino-verify-20261004').iterdir())
(root/'results/final-complete.json').write_text(json.dumps(dict(complete=True,raw_trace_remaining=0))+'\n')
