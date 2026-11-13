"""Deeper controls and functional combination, in a separate hook build."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
py=sys.executable
env=dict(os.environ,NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
workload=[py,'benchmarks/qwen_workload.py','--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
 '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
 '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
 '--requests','3','--warmup','2','--output-tokens','32']

def run(name,conditions,rounds=7,diagnostic=False,records=False):
    argv=[py,'benchmarks/run_ablation_matrix.py','--output','results/root-cause/'+name,
          '--spec','benchmarks/ablation_specs.json','--rounds',str(rounds),'--conditions',*conditions,
          '--compiler-python',str(root/'benchmarks/cache_python.py')]
    cmd=list(workload)
    if diagnostic:
        argv+=['--diagnostic'];cmd[cmd.index('--requests')+1]='1'
    local=dict(env)
    if records:local['NEUTRINO_RECORD_VALIDITY']='1'
    print('EXTRA START',name,flush=True)
    subprocess.run([*argv,'--',*cmd],cwd=root,env=local,check=True)
    print('EXTRA DONE',name,flush=True)

run('diagnostic-protocol',['S1','S4','S6','S3','S5','O0','O1'],1,True)
run('protocol-7round',['S1','S4','S6'])
run('blocking-pool-7round',['S3','S5'])
run('combined-7round',['O0','O1'])
run('map-validity',['V0'],1,True,True)
(root/'results/root-cause/extra-complete.json').write_text(json.dumps({'complete':True})+'\n')
