"""Post-diagnostic correction: orphan Events and effective size-class capacity."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True)
prior=Path('/root/autodl-tmp/neutrino-ring-output-20261003')
while not (prior/'results/complete.json').exists():time.sleep(2)
py=sys.executable
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
    NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
subprocess.run([py,'neutrino/build.py'],cwd=root,env=env,check=True)
subprocess.run(['cc','-shared','-fPIC','-O3','benchmarks/profile_preload.c','-ldl','-o','neutrino/build/profile-preload.so'],cwd=root,env=env,check=True)
shutil.copyfile(prior/'benchmarks/deep_cupti.so',root/'benchmarks/deep_cupti.so')
original=json.loads((root/'benchmarks/bulk_specs.json').read_text())
specs={c:original[c] for c in ['N','T8','LG64']};specs['LG64']['root']=str(prior)
specs['FO']=dict(mode='files',env={'NEUTRINO_BULK':'1'},root=str(prior))
specs['F']=dict(mode='files',env={'NEUTRINO_BULK':'1'})
specs['FB']=dict(mode='files',env={'NEUTRINO_BULK':'1','NEUTRINO_BULK_BATCH':'64'})
specs['FQ']=dict(mode='files',env={'NEUTRINO_BULK':'1','NEUTRINO_BULK_SMALL_HOSTS':'32','NEUTRINO_BULK_SMALL_DEVICES':'32'})
specs['FBQ']=dict(mode='files',env={**specs['FQ']['env'],'NEUTRINO_BULK_BATCH':'64'})
sp=out/'correction-specs.json';sp.write_text(json.dumps(specs,indent=2)+'\n')
work=[py,str(root/'benchmarks/qwen_capture_workload.py'),'--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
    '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin','--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
    '--requests','3','--warmup','2','--output-tokens','32']
def matrix(name,conditions,rounds=5,diag=False,records=False,requests=3,spec=sp):
    cmd=[py,'benchmarks/run_bulk_matrix.py','--output',str(out/name),'--spec',str(spec),'--conditions',*conditions,
        '--rounds',str(rounds),'--compiler-python',str(root/'benchmarks/cache_python.py')]
    if diag:cmd+=['--diagnostic']
    w=list(work);w[w.index('--requests')+1]=str(requests)
    e=dict(env)
    if records:e['BULK_RECORD_VALIDITY']='1'
    print('CORRECTION START',name,flush=True);subprocess.run([*cmd,'--',*w],cwd=root,env=e,check=True)
    print('CORRECTION DONE',name,flush=True)
matrix('validation',['F','FB','FQ','FBQ'],1,True,True,1)
matrix('main-5round',list(specs))
profile={c:{**specs[c],'env':{**specs[c]['env'],'DEEP_CUPTI':'1','DEEP_CUPTI_ACTIVITY':'1','NEUTRINO_PROFILE_COMPAT':'1'}} for c in ['FO','F','FB','FQ','FBQ']}
pp=out/'profile-specs.json';pp.write_text(json.dumps(profile,indent=2)+'\n')
matrix('cupti-activity',list(profile),1,False,False,1,pp)
runs=json.loads((out/'main-5round/runs.json').read_text());eligible=['F','FB','FQ','FBQ','LG64']
means={c:sum(r['wall_ns']/len(r['requests']) for r in runs if r['condition']==c)/5 for c in eligible};best=min(means,key=means.get)
(out/'selection.json').write_text(json.dumps(dict(selected=best,means_ns=means,rule='same-batch mean; independent confirmation only if a new candidate beats frozen LG64'),indent=2)+'\n')
if best!='LG64':
    matrix('final-anchor-7round',['N','T8',best],7)
    matrix('sustained-3round',['N','T8',best],3,requests=20)
subprocess.run([py,'tests/run_gpu_validation.py','--output',str(out/'default-regression'),'--trace-root','/dev/shm/neutrino-ablation'],cwd=root,env=env,check=True)
(out/'complete.json').write_text(json.dumps(dict(complete=True,selected=best))+'\n')
print('CORRECTION COMPLETE',flush=True)
