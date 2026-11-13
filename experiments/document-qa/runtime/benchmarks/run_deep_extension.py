"""Additional finite transfer/correlation diagnostics; GPU work is serial."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_deep_suite import run, root, out

while not (out/'diagnostics-complete.json').exists():time.sleep(2)
for c in ['D0','D3']:run('cupti-activity-paired',c)
for enabled in (0,1):
    for mode,name in [(0,'fresh_pageable'),(1,'reused_pageable'),(3,'pinned')]:
        dest=out/'d2h-correlation'/f'{name}-activity{enabled}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        env=dict(os.environ,DEEP_OUTPUT=str(dest/'cupti.json'),DEEP_CUPTI_ACTIVITY=str(enabled))
        command=[str(root/'benchmarks/d2h_cupti_micro'),'64','1','9723904',str(mode)]
        print('CORRELATION START',name,enabled,flush=True)
        result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,check=True,timeout=120)
        workload=json.loads(result.stdout);d=json.loads((dest/'cupti.json').read_text())
        assert workload['full_validation'] and d['driver_callbacks']==64 and not d['overflow'] and not d['dropped']
        if enabled:assert d['copy_activity_bytes']==64*9723904 and d['unmatched_copies']==0
        (dest/'workload.json').write_text(json.dumps(workload,indent=2)+'\n')
        (dest/'complete.json').write_text(json.dumps({'passed':True,'command':command})+'\n')
        print('CORRELATION DONE',name,enabled,flush=True)
print('NUMA MICRO START',flush=True)
with (out/'d2h-numa.json').open('w') as fp:
    subprocess.run([str(root/'benchmarks/d2h_numa_micro'),'7'],cwd=root,stdout=fp,check=True,timeout=600)
(out/'extension-complete.json').write_text(json.dumps({'complete':True})+'\n')
print('EXTENSION COMPLETE',flush=True)
