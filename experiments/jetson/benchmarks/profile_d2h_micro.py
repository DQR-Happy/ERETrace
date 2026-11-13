"""Profile a completed standalone D2H buffer after all model experiments."""
import json
import os
from pathlib import Path
import subprocess
import time
from nsight_sqlite import summarize

root=Path(__file__).resolve().parents[1]
out=root/'results/root-cause'
while not (out/'extra-complete.json').exists():
    time.sleep(2)
nsys='/root/autodl-tmp/neutrino-profile-tools/nsys/opt/nvidia/nsight-systems/2024.2.3/bin/nsys'
dest=out/'d2h-profile';dest.mkdir(exist_ok=True)
for name,mode in [('fresh_pageable',0),('reused_pageable',1),('reused_pinned',3)]:
    run=dest/name;run.mkdir(exist_ok=True)
    if (run/'summary.json').exists(): continue
    cmd=[nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none',
         '--capture-range=cudaProfilerApi','--capture-range-end=stop',
         '--force-overwrite=true','--output='+str(run/'capture'),
         str(root/'benchmarks/d2h_profile_micro'),'64','1','9723904',str(mode)]
    (run/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    print('D2H PROFILE START',name,flush=True)
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True,timeout=120)
    line=next(line for line in result.stdout.splitlines() if line.startswith('{"loops"'))
    workload=json.loads(line)
    assert workload['full_validation'] and len(workload['results'])==1
    (run/'workload.json').write_text(json.dumps(workload,indent=2)+'\n')
    subprocess.run([nsys,'export','--type=sqlite','--force-overwrite=true',
                    '--output='+str(run/'capture.sqlite'),str(run/'capture.nsys-rep')],
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    summary=summarize(run/'capture.sqlite',workload['results'][0]['wall_ns'])
    copies=summary['gpu']['CUPTI_ACTIVITY_KIND_MEMCPY']['by_copy_kind']['2']
    assert copies['bytes']==64*9723904
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for pattern in ('*.nsys-rep','*.sqlite','*.qdstrm'):
        for path in run.glob(pattern): path.unlink()
    print('D2H PROFILE DONE',name,'driver API:',summary['driver_api_available'],flush=True)
(dest/'complete.json').write_text(json.dumps({'complete':True})+'\n')
