import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];py=sys.executable
v2=Path('/root/autodl-tmp/neutrino-nvcomp-pipeline-20261004');v3=Path('/root/autodl-tmp/neutrino-hotpath-20261004')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def run(r,name,conds,rounds,*extra):
 if (r/'results'/name/'complete.json').exists():return
 print(r.name,name,'START',flush=True)
 with (r/(name+'.log')).open('w') as f:subprocess.run([py,'benchmarks/run_codec_comparison_20261004.py','--name',name,'--conditions',*conds.split(),'--rounds',str(rounds),*extra],cwd=r,env=env,stdout=f,stderr=f,check=True)
 print(name,'DONE',flush=True)
assert (v2/'results/nvpipe-screen/complete.json').exists()
run(v2,'nvpipe-confirmation','N KOLD K16 E NS',7)
run(v2,'nvpipe-host','K16 E NA NS NS256',3,'--requests','1','--diagnostic','--layout')
with (v2/'nvpipe-memcheck.log').open('w') as f:subprocess.run(['compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'benchmarks/gpu_nvpipe_smoke.py','benchmarks/nvcomp_pipeline.so','results/nvpipe-memcheck.json'],cwd=v2,env=env,stdout=f,stderr=f,check=True)
run(v3,'hotpath-validation','KAH EAH',1,'--requests','1','--tokens','8','--validate')
run(v3,'hotpath-confirmation','N KOLD K16 E KA KH KAH EAH',7)
run(v3,'hotpath-host','K16 E KA KH KAH EAH',3,'--requests','1','--diagnostic')
(v3/'results/confirmation-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
