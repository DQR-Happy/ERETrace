"""Corrected copy association controls and paired final/native anchor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];out=root/'results/deep';py=sys.executable
while not (out/'cupti-v2-complete.json').exists():time.sleep(2)
for enabled in (0,1):
    for mode,name in [(0,'fresh_pageable'),(1,'reused_pageable'),(3,'pinned')]:
        dest=out/'d2h-correlation-v2'/f'{name}-activity{enabled}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        env=dict(os.environ,DEEP_OUTPUT=str(dest/'cupti.json'),DEEP_CUPTI_ACTIVITY=str(enabled))
        command=[str(root/'benchmarks/d2h_cupti_micro'),'64','1','9723904',str(mode)]
        p=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,check=True,timeout=120)
        w=json.loads(p.stdout);d=json.loads((dest/'cupti.json').read_text())
        assert w['full_validation'] and d['driver_callbacks']==64 and not d['overflow'] and not d['dropped']
        assert d['correlation_schema']==2 and not d['ambiguous_copies']
        if enabled:assert d['copy_activity_bytes']==64*9723904 and not d['unmatched_copies']
        (dest/'workload.json').write_text(json.dumps(w,indent=2)+'\n')
        (dest/'complete.json').write_text(json.dumps(dict(passed=True,command=command))+'\n')
        print('COPY V2 DONE',name,enabled,flush=True)
workload=[py,'benchmarks/qwen_workload.py','--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
 '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
 '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
 '--requests','3','--warmup','2','--output-tokens','32']
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
 NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
dest=out/'final-anchor-7round'
if not (dest/'complete.json').exists():
    command=[py,'benchmarks/run_ablation_matrix.py','--output',str(dest),'--spec',str(out/'stream-specs.json'),
     '--conditions','T0','T1','T8','--rounds','7','--compiler-python',str(root/'benchmarks/cache_python.py'),'--',*workload]
    subprocess.run(command,cwd=root,env=env,check=True)
(out/'final-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
print('DEEP FINAL COMPLETE',flush=True)
