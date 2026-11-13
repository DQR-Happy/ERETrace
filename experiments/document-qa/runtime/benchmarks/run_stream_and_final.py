"""Separate-copy-stream paired experiment, followed by corrected syscall diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];py=sys.executable;out=root/'results/deep'
prior=Path('/root/autodl-tmp/neutrino-deep-async-20261003/results/deep')
while not (prior/'async-complete.json').exists():
    pid=json.loads((prior/'async-pid.json').read_text())['pid']
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError('Async experiment did not complete')
    time.sleep(2)
specs=json.loads((prior/'async-specs.json').read_text())
specs['T7']=dict(mode='files',env={**specs['T4']['env'],'NEUTRINO_ABLATION_ASYNC_COPY_STREAM':'1'})
specs['T8']=dict(mode='files',env={**specs['T5']['env'],'NEUTRINO_ABLATION_ASYNC_COPY_STREAM':'1'})
specpath=out/'stream-specs.json';specpath.write_text(json.dumps(specs,indent=2)+'\n')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
         NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
workload=[py,'benchmarks/qwen_workload.py','--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
 '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
 '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
 '--requests','3','--warmup','2','--output-tokens','32']
for name,diag,rounds in [('stream-diagnostic',True,1),('stream-7round',False,7)]:
    print('STREAM START',name,flush=True)
    w=list(workload);cmd=[py,'benchmarks/run_ablation_matrix.py','--output',str(out/name),
        '--spec',str(specpath),'--conditions','T4','T5','T7','T8','--rounds',str(rounds),
        '--compiler-python',str(root/'benchmarks/cache_python.py')]
    if diag:cmd+=['--diagnostic'];w[w.index('--requests')+1]='1'
    subprocess.run([*cmd,'--',*w],cwd=root,env=env,check=True)
    print('STREAM DONE',name,flush=True)
import run_deep_suite as diagnostics
diagnostics.specs.update(specs)
for condition in ['T4','T7']:diagnostics.run('cupti-activity-stream',condition)
for condition in ['S0','S1','S6','S3','F0','F1','F2','F4','O1']:diagnostics.run('syscalls-v2',condition)
subprocess.run([py,'tests/run_gpu_validation.py','--output',str(out/'final-default-regression'),
                '--trace-root','/dev/shm/neutrino-ablation'],cwd=root,env=env,check=True)
(out/'stream-complete.json').write_text(json.dumps({'complete':True})+'\n')
print('STREAM AND FINAL COMPLETE',flush=True)
