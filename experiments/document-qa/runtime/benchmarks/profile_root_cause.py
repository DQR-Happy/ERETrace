"""Finite CUDA profiling after the unprofiled suite; retain compact summaries only."""
import collections
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
out=root/'results/root-cause'
while not (out/'suite-complete.json').exists():
    pid=json.loads((out/'suite-pid.json').read_text())['pid']
    try: os.kill(pid,0)
    except ProcessLookupError: raise RuntimeError('Suite stopped before completion')
    time.sleep(2)
nsys=Path('/root/autodl-tmp/neutrino-profile-tools/nsys/opt/nvidia/nsight-systems/2024.2.3/bin/nsys')
py=sys.executable
dest=out/'nsight';dest.mkdir(exist_ok=True)
cases={'native':None,'framework':'framework','probe':'probe','files':'files','framework_async':'framework'}


def union(intervals):
    total=0;last=None
    for a,b in sorted(intervals):
        if last is None: last=[a,b]
        elif a<=last[1]: last[1]=max(last[1],b)
        else: total+=last[1]-last[0];last=[a,b]
    return total+(last[1]-last[0] if last else 0)


def summarize(path,wall_ns):
    conn=sqlite3.connect(path)
    tables={r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    names=dict(conn.execute('select id,value from StringIds'))
    schema={t:[r[1] for r in conn.execute(f'pragma table_info({t})')] for t in tables if t.startswith('CUPTI_ACTIVITY_KIND')}
    activities=[];gpu={};kernels={};copies=[]
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table not in tables: continue
        columns=schema[table]
        rows=[dict(zip(columns,row)) for row in conn.execute(f'select * from {table}')]
        activities += [(r['start'],r['end']) for r in rows]
        gpu[table]=dict(count=len(rows),sum_ns=sum(r['end']-r['start'] for r in rows),
                        union_ns=union([(r['start'],r['end']) for r in rows]))
        if table.endswith('MEMCPY'):
            by_kind=collections.defaultdict(lambda:dict(count=0,bytes=0,sum_ns=0))
            for r in rows:
                g=by_kind[str(r['copyKind'])];g['count']+=1;g['bytes']+=r['bytes'];g['sum_ns']+=r['end']-r['start']
            gpu[table]['by_copy_kind']=dict(by_kind);copies=rows
        if table.endswith('KERNEL'):
            for r in rows:
                name=names[r.get('demangledName',r.get('shortName'))]
                g=kernels.setdefault(name,dict(count=0,sum_ns=0));g['count']+=1;g['sum_ns']+=r['end']-r['start']
    apis={};driver=[]
    for table in ('CUPTI_ACTIVITY_KIND_DRIVER','CUPTI_ACTIVITY_KIND_RUNTIME'):
        if table not in tables: continue
        rows=[dict(zip(schema[table],row)) for row in conn.execute(f'select * from {table}')]
        agg={}
        for r in rows:
            name=names[r['nameId']];r['name']=name
            g=agg.setdefault(name,dict(count=0,sum_ns=0));g['count']+=1;g['sum_ns']+=r['end']-r['start']
        apis[table]=agg
        if table.endswith('DRIVER'):driver=sorted(rows,key=lambda r:r['start'])
    gaps=[];d2h_api={r['correlationId']:r for r in driver if r['name'].startswith('cuMemcpyDtoH')}
    correlated=[]
    for r in copies:
        api=d2h_api.get(r['correlationId'])
        if api: correlated.append(dict(bytes=r['bytes'],api_ns=api['end']-api['start'],gpu_ns=r['end']-r['start'],
                                      prefix_ns=r['start']-api['start'],suffix_ns=api['end']-r['end']))
    for a,b in zip(driver,driver[1:]):
        if a['name'].startswith('cuMemcpyDtoH') and b['name'].startswith('cuMemFree') and a['globalTid']==b['globalTid']:
            gaps.append(b['start']-a['end'])
    correlated_by_size={}
    for r in correlated:
        g=correlated_by_size.setdefault(str(r['bytes']),dict(count=0,api_ns=0,gpu_ns=0,prefix_ns=0,suffix_ns=0))
        g['count']+=1
        for k in ('api_ns','gpu_ns','prefix_ns','suffix_ns'):g[k]+=r[k]
    bounds=[min(a for a,b in activities),max(b for a,b in activities)] if activities else [0,0]
    result=dict(schema=schema,gpu=gpu,kernels=kernels,apis=apis,
        profiled_wall_ns=wall_ns,gpu_activity_union_ns=union(activities),gpu_activity_span_ns=bounds[1]-bounds[0],
        d2h_correlated_by_size=correlated_by_size,
        d2h_to_next_free_gap=dict(count=len(gaps),sum_ns=sum(gaps),min_ns=min(gaps) if gaps else 0,max_ns=max(gaps) if gaps else 0))
    conn.close();return result


for case,mode in cases.items():
    run=dest/case
    if (run/'summary.json').exists():
        continue
    if run.exists():
        failed=dest/(case+'-failed-initial')
        run.rename(failed)
    print(f'PROFILE START {case}',flush=True)
    run.mkdir();trace=Path('/dev/shm/neutrino-ablation')/f'profile-{case}';trace.mkdir(exist_ok=False)
    env=dict(os.environ,NEUTRINO_BENCH_DIR=str(run),NEUTRINO_PROFILE='1',NEUTRINO_PROFILE_COMPAT='1',
             NEUTRINO_ABLATION_STATS='1',NEUTRINO_COMPILER_CACHE_ROOT=str(root/'compiler-cache'))
    if case=='framework_async':env['NEUTRINO_ABLATION_NO_EVENTS']='1'
    command=[py,'benchmarks/qwen_workload.py','--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
        '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
        '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
        '--requests','1','--warmup','2','--output-tokens','32']
    if mode:
        command=[py,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null',
                 '--tracedir',str(trace),'--ablation-mode',mode,'--python',str(root/'benchmarks/cache_python.py'),*command]
    cmd=[str(nsys),'profile','--force-overwrite=true','--trace=cuda',
         '--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop',
         '--output='+str(run/'capture'),*command]
    (run/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    try:
        with (run/'profile.log').open('w') as log:
            subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=log,timeout=600,check=True)
        w=json.loads((run/'workload.json').read_text())
        with (run/'export.log').open('w') as log:
            subprocess.run([str(nsys),'export','--type=sqlite','--force-overwrite=true',
                            '--output='+str(run/'capture.sqlite'),str(run/'capture.nsys-rep')],stdout=log,stderr=log,check=True)
        result=summarize(run/'capture.sqlite',w['wall_ns'])
        if mode:
            d=json.loads((run/'measurement-ablation.json').read_text())
            assert sum(g['calls'] for g in d['groups'])==16648
            assert not d['cold'] and not any(g['fallback'] or g['errors'] for g in d['groups'])
        assert result['gpu'].get('CUPTI_ACTIVITY_KIND_KERNEL',{}).get('count',0)>=16648
        (run/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        # The schema, counts, correlated sums and configuration are retained; raw activity is temporary.
        for pattern in ('*.nsys-rep','*.sqlite','*.qdstrm'):
            for p in run.glob(pattern):p.unlink()
        print(f'PROFILE DONE {case}',flush=True)
    finally:
        shutil.rmtree(trace)
(dest/'complete.json').write_text(json.dumps(dict(complete=True))+'\n')
