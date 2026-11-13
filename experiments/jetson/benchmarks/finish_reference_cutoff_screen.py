"""Run only after V1 finishes, so builds never perturb V1 performance."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];py=sys.executable;out=root/'results';out.mkdir(exist_ok=True)
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def call(cmd,log):
 print('START',log,flush=True)
 with (root/log).open('w') as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=f,check=True)
 print('DONE',log,flush=True)
call([py,'-m','neutrino.build'],'build-cutoff.log')
for name,conditions,rounds,args in [('cutoff-validation','BP K32 K16 K8 K4 K4F',1,['--requests','1','--validate']),('cutoff-screen','N BOLD BP K32 K16 K8 K4',3,['--requests','1'])]:
 call([py,'benchmarks/run_reference_20261004.py','--name',name,'--conditions',*conditions.split(),'--rounds',str(rounds),*args],name+'.log')
(out/'screen-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
