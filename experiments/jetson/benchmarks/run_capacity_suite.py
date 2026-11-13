import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arch',default='sm_86');p.add_argument('--output',default='results/capacity-main');p.add_argument('--rounds',type=int,default=7);a=p.parse_args()
root=Path(__file__).resolve().parents[1];env=dict(os.environ,PYTHONPATH=str(root),PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
for r in range(1,a.rounds+1):
    args=[sys.executable,'benchmarks/core_capacity_experiment.py','--output',a.output,'--round',str(r),'--arch',a.arch]+(['--build'] if r==1 else [])
    print('ROUND',r,flush=True);subprocess.run(args,cwd=root,env=env,check=True)
(root/a.output/'complete.json').write_text('{"rounds":'+str(a.rounds)+',"passed":true}\n')
