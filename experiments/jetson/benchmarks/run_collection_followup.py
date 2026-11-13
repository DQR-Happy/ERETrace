"""Serialized follow-up on an isolated build; never overlaps the first matrix."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

resume='--resume' in sys.argv[1:]
if any(x!='--resume' for x in sys.argv[1:]):raise SystemExit('Only --resume is accepted')

root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable
prior=Path('/root/autodl-tmp/neutrino-bulk-20261003/results/main-7round/complete.json')
while not prior.exists():time.sleep(2)
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
         NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
subprocess.run([py,'neutrino/build.py'],cwd=root,env=env,check=True)
subprocess.run(['cc','-shared','-fPIC','-O3','benchmarks/profile_preload.c','-ldl','-o','neutrino/build/profile-preload.so'],cwd=root,env=env,check=True)
subprocess.run(['nvcc','-O3','-arch=sm_86','--cubin','benchmarks/capture_pack.cu','-o','benchmarks/capture_pack.cubin'],cwd=root,env=env,check=True)
subprocess.run(['nvcc','-O3','-arch=sm_86','benchmarks/clock_domains.cu','-o','benchmarks/clock_domains'],cwd=root,env=env,check=True)
with (out/'clock-domains.json').open('w') as fp:subprocess.run([str(root/'benchmarks/clock_domains')],stdout=fp,cwd=root,env=env,check=True)
subprocess.run([py,'benchmarks/gpu_codec_smoke.py',str(root/'benchmarks/capture_pack.cubin'),str(out/'gpu-codec-smoke.json')],cwd=root,env=env,check=True)
specs=json.loads((root/'benchmarks/bulk_specs.json').read_text());specpath=root/'benchmarks/bulk_specs.json'
workload=[py,str(root/'benchmarks/qwen_capture_workload.py'),'--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
 '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
 '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
 '--requests','3','--warmup','2','--output-tokens','32']
def matrix(name,conditions,rounds=7,diag=False,requests=3,records=False,spec=specpath):
    if resume and (out/name/'complete.json').exists():
        saved=json.loads((out/name/'config.json').read_text())
        assert saved['conditions']==conditions and saved['rounds']==rounds and saved['diagnostic']==diag
        assert saved['command'][saved['command'].index('--requests')+1]==str(requests)
        print('FOLLOWUP VERIFIED COMPLETE',name,flush=True);return
    command=[py,'benchmarks/run_bulk_matrix.py','--output',str(out/name),'--spec',str(spec),
        '--conditions',*conditions,'--rounds',str(rounds),'--compiler-python',str(root/'benchmarks/cache_python.py')]
    w=list(workload);w[w.index('--requests')+1]=str(requests)
    if diag:command+=['--diagnostic']
    local=dict(env)
    if records:local['BULK_RECORD_VALIDITY']='1'
    print('FOLLOWUP START',name,flush=True);subprocess.run([*command,'--',*w],cwd=root,env=local,check=True)
    print('FOLLOWUP DONE',name,flush=True)
matrix('ring-validation',['LGV','LR','LP','TZ','T64'],1,True,1,True)
matrix('ring-main-7round',['N','T8','L','TZ','T16','T64','LR','LR64','LP','LG','LG64','LGZ'])
matrix('final-diagnostic',['T8','L','TZ','T64','LR','LP','LG','LG64','LGZ','C','B64','S64','D2','P','Z','HC'],1,True,1)
prof={}
for name in ['T8','TZ','T64','C','B64','HC','D2','LR','LP','LG','LG64']:
    prof[name]={**specs[name],'env':{**specs[name]['env'],'DEEP_CUPTI':'1','DEEP_CUPTI_ACTIVITY':'1','NEUTRINO_PROFILE_COMPAT':'1'}}
profpath=out/'profile-specs.json';profpath.write_text(json.dumps(prof,indent=2)+'\n')
matrix('cupti-activity',list(prof),1,False,1,spec=profpath)
runs=json.loads((out/'ring-main-7round/runs.json').read_text())
eligible=['L','TZ','T64','LP','LG','LG64','LGZ']
mean={c:sum(r['wall_ns']/len(r['requests']) for r in runs if r['condition']==c)/7 for c in eligible}
best=min(mean,key=mean.get)
(out/'selection.json').write_text(json.dumps(dict(rule='smallest seven-round mean among complete file outputs; RAM excluded',means_ns=mean,selected=best),indent=2)+'\n')
matrix('final-anchor-7round',['N','T8',best])
matrix('sustained-3round',['N','T8',best],3,requests=20)
if not (resume and (out/'default-regression/summary.json').exists()):
    subprocess.run([py,'tests/run_gpu_validation.py','--output',str(out/'default-regression'),'--trace-root','/dev/shm/neutrino-ablation'],cwd=root,env=env,check=True)
print('COLLECTION FOLLOWUP COMPLETE',flush=True)
(out/'complete.json').write_text(json.dumps(dict(complete=True,selected=best))+'\n')
