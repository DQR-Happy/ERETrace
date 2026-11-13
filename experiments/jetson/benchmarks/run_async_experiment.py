"""Wait for mechanism tests, then validate and measure full asynchronous capture."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_ablation_matrix import trace_check

root=Path(__file__).resolve().parents[1]
py=sys.executable
out=root/'results/deep';out.mkdir(parents=True,exist_ok=True)
prior=Path('/root/autodl-tmp/neutrino-deep-20261003/results/deep')
while not (prior/'extension-complete.json').exists():
    pid=json.loads((prior/'extension-pid.json').read_text())['pid']
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError('Mechanism extension did not complete')
    time.sleep(2)
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
         NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
specs=json.loads((root/'benchmarks/ablation_specs.json').read_text())
specs.update(T0=dict(mode=None,env={}),T1=specs['O0'],T2=specs['O1'])
for name,slots in [('T3',1),('T4',8),('T5',32),('T6',8)]:
    specs[name]=dict(mode='files',env={'NEUTRINO_ABLATION_ASYNC':'1','NEUTRINO_ABLATION_ASYNC_SLOTS':str(slots)})
specs['T6']['env']['NEUTRINO_ABLATION_ASYNC_BLOCKING']='1'
specs['VA']=dict(mode='files',env={**specs['V0']['env'],**specs['T4']['env']})
specpath=out/'async-specs.json';specpath.write_text(json.dumps(specs,indent=2)+'\n')
workload=[py,'benchmarks/qwen_workload.py','--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
 '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
 '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
 '--requests','3','--warmup','2','--output-tokens','32']
def matrix(name,conditions,rounds=7,diag=False,records=False):
    cmd=[py,'benchmarks/run_ablation_matrix.py','--output',str(out/name),'--spec',str(specpath),
         '--conditions',*conditions,'--rounds',str(rounds),'--compiler-python',str(root/'benchmarks/cache_python.py')]
    w=list(workload)
    if diag:cmd+=['--diagnostic'];w[w.index('--requests')+1]='1'
    local=dict(env)
    if records:local['NEUTRINO_RECORD_VALIDITY']='1'
    print('ASYNC MATRIX START',name,flush=True)
    subprocess.run([*cmd,'--',*w],cwd=root,env=local,check=True)
    print('ASYNC MATRIX DONE',name,flush=True)
matrix('async-diagnostic',['T0','T1','T2','T3','T4','T5','T6'],1,True)
matrix('async-record-validity',['VA'],1,True,True)
matrix('async-7round',['T0','T1','T2','T3','T4','T5','T6'])
print('ASYNC DEFAULT REGRESSION START',flush=True)
subprocess.run([py,'tests/run_gpu_validation.py','--output',str(out/'default-regression'),
                '--trace-root','/dev/shm/neutrino-ablation'],cwd=root,env=env,check=True)
(out/'async-complete.json').write_text(json.dumps({'complete':True})+'\n')
print('ASYNC COMPLETE',flush=True)
