"""Independent confirmation of all screened cutoffs, followed by diagnostics."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable
assert (out/'screen-complete.json').exists()
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def matrix(name,conditions,rounds,*args):
 if (out/name/'complete.json').exists():return
 print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as f:subprocess.run([py,'benchmarks/run_reference_20261004.py','--name',name,'--conditions',*conditions.split(),'--rounds',str(rounds),*args],cwd=root,env=env,stdout=f,stderr=f,check=True)
 print('DONE',name,flush=True)
matrix('cutoff-confirmation-7round','N BOLD BP K32 K16 K8 K4',7)
matrix('cutoff-host-3round','BP K32 K16 K4',3,'--diagnostic','--requests','1','--layout')
matrix('cutoff-gpu-3round','N BP K16 K4',3,'--gpu','--requests','1')
matrix('compat-control-3round','BP PC',3,'--requests','1')
(out/'confirmation-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
