"""Build the new isolated cutoff variant only after the prior suite completes."""
import os,shutil,subprocess,sys,time
from pathlib import Path
base=Path('/root/autodl-tmp/neutrino-reference-20261004');root=Path('/root/autodl-tmp/neutrino-reference-cutoff-20261004');stage=Path('/root/autodl-tmp/neutrino-reference-cutoff-stage')
deadline=time.monotonic()+7200;absent=0
while not (base/'results/suite-complete.json').exists():
 alive=subprocess.run(['pgrep','-f','[f]inish_reference_20261004.py'],stdout=subprocess.DEVNULL).returncode==0
 absent=0 if alive else absent+1
 if absent>6 or time.monotonic()>deadline:raise RuntimeError('Previous suite incomplete; do not overlap or silently continue')
 time.sleep(5)
root.mkdir(exist_ok=False)
subprocess.run(['rsync','-a','--exclude=results','--exclude=vendor','--exclude=upload-layout-error','--exclude=__pycache__','--exclude=*.log',str(base)+'/',str(root)+'/'],check=True)
for p in (stage/'core').iterdir():shutil.copy2(p,root/'neutrino/src'/p.name)
for p in (stage/'benchmarks').iterdir():shutil.copy2(p,root/'benchmarks'/p.name)
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
with (root/'cutoff-screen-suite.log').open('w') as f:subprocess.run([sys.executable,'benchmarks/finish_reference_cutoff_screen.py'],cwd=root,env=env,stdout=f,stderr=f,check=True)
