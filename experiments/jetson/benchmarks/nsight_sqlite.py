"""Compact CUDA activity and correlation summary; handles one-to-many copies."""
import collections
import sqlite3

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
            by_size={}
            for r in rows:
                key=f"{r['copyKind']}:{r['bytes']}"
                g=by_size.setdefault(key,dict(count=0,bytes_each=r['bytes'],copy_kind=r['copyKind'],sum_ns=0))
                g['count']+=1;g['sum_ns']+=r['end']-r['start']
            gpu[table]['by_size']=by_size
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
    pieces=collections.defaultdict(list)
    for r in copies: pieces[r['correlationId']].append(r)
    for cid,part in pieces.items():
        api=d2h_api.get(cid)
        if api: correlated.append(dict(bytes=sum(r['bytes'] for r in part),api_ns=api['end']-api['start'],
            gpu_ns=union([(r['start'],r['end']) for r in part]),
            prefix_ns=min(r['start'] for r in part)-api['start'],suffix_ns=api['end']-max(r['end'] for r in part)))
    for a,b in zip(driver,driver[1:]):
        if a['name'].startswith('cuMemcpyDtoH') and b['name'].startswith('cuMemFree') and a['globalTid']==b['globalTid']:
            gaps.append(b['start']-a['end'])
    correlated_by_size={}
    for r in correlated:
        g=correlated_by_size.setdefault(str(r['bytes']),dict(count=0,api_ns=0,gpu_ns=0,prefix_ns=0,suffix_ns=0))
        g['count']+=1
        for k in ('api_ns','gpu_ns','prefix_ns','suffix_ns'):g[k]+=r[k]
    bounds=[min(a for a,b in activities),max(b for a,b in activities)] if activities else [0,0]
    result=dict(schema=schema,gpu=gpu,kernels=kernels,apis=apis,driver_api_available=bool(driver),
        profiled_wall_ns=wall_ns,gpu_activity_union_ns=union(activities),gpu_activity_span_ns=bounds[1]-bounds[0],
        d2h_correlated_by_size=correlated_by_size,
        d2h_to_next_free_gap=dict(count=len(gaps),sum_ns=sum(gaps),min_ns=min(gaps) if gaps else 0,max_ns=max(gaps) if gaps else 0))
    conn.close();return result

