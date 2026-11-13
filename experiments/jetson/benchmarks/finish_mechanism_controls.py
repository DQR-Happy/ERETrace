import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable
while not (out/'suite-complete.json').exists():time.sleep(3)
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def call(cmd,name,extra=None):
 print('START',name,flush=True)
 with (root/name).open('w') as f:subprocess.run(cmd,cwd=root,env={**env,**(extra or {})},stdout=f,stderr=f,check=True)
 print('DONE',name,flush=True)
lib='/root/autodl-tmp/neutrino-mechanism-20261004/vendor/nvidia/libnvcomp'
call(['nvcc','-std=c++17','-arch=sm_86','-O3','benchmarks/hier_service_micro.cu','-I'+lib+'/include','-L'+lib+'/lib64','-Xlinker','-rpath','-Xlinker',lib+'/lib64','-l:libnvcomp.so.5','-o','benchmarks/hier_service_micro'],'hier-service-build.log')
for name,extra in [('reference-domains',{'MECHANISM_RANGES_ONLY':'1'}),('hier-service-control',{'MECHANISM_HIER_MICRO':'1'})]:
 if (out/name/'complete.json').exists():continue
 call([py,'benchmarks/run_mechanism_20261004.py','--name',name,'--conditions','DW','--rounds','1','--requests','1','--diagnostic'],name+'.log',{'MECHANISM_CODEC_CONTROL':'1',**extra})
call([py,'benchmarks/verification_provenance.py'],'provenance-final.log')
assert not any(Path('/dev/shm/neutrino-mechanism-20261004').iterdir())
(out/'controls-complete.json').write_text(json.dumps(dict(complete=True,raw_trace_remaining=0))+'\n')
