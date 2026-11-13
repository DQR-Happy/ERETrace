import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];py=sys.executable;out=root/'results'
assert json.loads((out/'reference-smoke.json').read_text())['passed']
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def matrix(name,conditions,rounds,*args):
 print('START',name,flush=True)
 with (root/(name+'.log')).open('w') as f:subprocess.run([py,'benchmarks/run_reference_20261004.py','--name',name,'--conditions',*conditions.split(),'--rounds',str(rounds),*args],cwd=root,env=env,stdout=f,stderr=f,check=True)
 print('DONE',name,flush=True)
matrix('reference-validation','H2 R0 T4 T8 T16 BP T4P R0F BPF T4PF',1,'--requests','1','--validate')
matrix('reference-screen','N HOLD H2 R0 T4 T8 T16 T32 BP T4P T8P',3,'--requests','1')
(out/'screen-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
