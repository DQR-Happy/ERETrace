"""Serial follow-up suite; starts only after the seven-round main matrix succeeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
out=root/'results/root-cause'
main=out/'main-7round'
while not (main/'complete.json').exists():
    pid=json.loads((out/'main-pid.json').read_text())['pid']
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        raise RuntimeError('Main matrix stopped before successful completion')
    time.sleep(2)

py=sys.executable
env=dict(os.environ,NEUTRINO_COMPILER_CACHE_ROOT=str(root/'compiler-cache'))
hardware=json.loads((root/'compiler-cache/hardware.json').read_text())
current=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version,compute_cap','--format=csv,noheader'],text=True).strip()
assert current==hardware['gpu'], 'Hardware fingerprint changed'
workload=[py,'benchmarks/qwen_workload.py','--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
 '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
 '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
 '--requests','3','--warmup','2','--output-tokens','32']

def matrix(name,conditions,rounds=7,diagnostic=False,spec=True):
    cmd=[py,'benchmarks/run_ablation_matrix.py','--output',str(out/name),
         '--rounds',str(rounds),'--conditions',*conditions,
         '--compiler-python',str(root/'benchmarks/cache_python.py')]
    if spec: cmd+=['--spec','benchmarks/ablation_specs.json']
    if diagnostic:
        cmd+=['--diagnostic']
        command=list(workload); command[command.index('--requests')+1]='1'
    else: command=workload
    cmd+=['--',*command]
    print(f'SUITE START {name}',flush=True)
    subprocess.run(cmd,cwd=root,env=env,check=True)
    print(f'SUITE DONE {name}',flush=True)

matrix('diagnostic-final',[f'P{i}' for i in range(9)],1,True,False)
matrix('diagnostic-variants',['K0','K1','K2','K3','S2','S3','S4','F1','F2','F3','F4','D1','D3','B1','B2','B3'],1,True)
matrix('targeted-7round',['K0','K1','K2','K3'])
matrix('sync-7round',['S0','S1','S2','S3','S4'])
matrix('files-7round',['F0','F1','F2','F3','F4'])
matrix('memory-7round',['D0','D1','D3','B0','B1','B2','B3'])
print('SUITE START d2h-micro',flush=True)
with (out/'d2h-micro.json').open('w') as fp:
    subprocess.run([str(root/'benchmarks/d2h_micro'),'64','7'],cwd=root,stdout=fp,env=env,check=True)
(out/'suite-complete.json').write_text(json.dumps({'complete':True})+'\n')
print('SUITE COMPLETE',flush=True)
