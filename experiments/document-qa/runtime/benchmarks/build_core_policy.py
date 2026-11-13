"""Freeze policies using only designated calibration artifacts, never holdouts."""
import argparse,hashlib,json,statistics
from pathlib import Path
from neutrino.probe.core_policy import digest_json,resource_choice,measured_choice

p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
p.add_argument('--resources',type=Path,required=True);p.add_argument('--compiler',type=Path,required=True)
p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1];a.output.mkdir(exist_ok=True)
sources={n:hashlib.sha256((root/'neutrino/probe'/n).read_bytes()).hexdigest() for n in ['cuda.py','core_codegen.py']}
res=json.loads(a.resources.read_text());compiled=json.loads(a.compiler.read_text())
resources={}
for r in res:
    if r['kind']!='probed' or r['mode'] not in ['shift','late','vector','late_vector']:continue
    k=r['kernel'];m=r['mode'];s=next(v for v in compiled if v['mode']==m and v['kernel']==k and v['version']=='probed')
    row=dict(registers=r['registers'],local_bytes=r['local_bytes'],residency=r['max_blocks_per_sm'],instructions=s['instructions'])
    old=resources.setdefault(k,{}).get(m)
    if old:
        row['residency']=min(old['residency'],row['residency'])
        row['local_bytes']=max(old['local_bytes'],row['local_bytes'])
    resources[k][m]=row
manifests={}
for f in a.cache.glob('*/manifest.json'):
    d=json.loads(f.read_text());key=d['key_data']
    if key['kernel'] in resources:manifests[key['kernel']]=key
timings={};train_files={};counts={}
for run in json.loads((a.training/'runs.json').read_text()):
    c=run['condition'];mode={'S':'shift','A':'late','V':'vector','AV':'late_vector'}[c]
    f=a.training/f"r{run['round']}-{c}"/'cupti.json';gpu=json.loads(f.read_text())
    assert not gpu['dropped'] and not gpu['overflow']
    capture=json.loads((f.parent/'capture-summary.json').read_text())['arena_engines'][0]
    assert capture['delta_fallbacks']==0
    train_files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for name in resources:
        matches=[k for k in gpu['kernels'] if k['name']==name or (len(k['name'])==511 and name.startswith(k['name']))]
        assert matches,name
        counts.setdefault(name,set()).add(sum(k['count'] for k in matches))
        timings.setdefault(name,{}).setdefault(mode,{})[run['round']]=sum(k['gpu_ns'] for k in matches)
assert all(len(v)==1 for v in counts.values())
times={n:{m:[v[i] for i in sorted(v)] for m,v in modes.items()} for n,modes in timings.items()}
policies={};decisions=[]
for label in ['G','R','T']:
    sample=next(iter(manifests.values()))
    policy=dict(schema=1,default='shift',hardware_digest=digest_json(json.loads((a.cache/'hardware.json').read_text())),compiler_sources=sources,compiler_tools=sample['tools'],functions={},training_files=train_files,method=label,threshold=.03)
    for name,r in resources.items():
        mode=measured_choice(times[name],r) if label=='T' else resource_choice(r,'registers' if label=='G' else 'resource')
        policy['functions'][name]=dict(mode=mode,original_sha256=manifests[name]['original'])
        decisions.append(dict(strategy=label,kernel=name,mode=mode,resources=r,training_ns=times[name],training_count=next(iter(counts[name]))))
    policies[label]=policy
    (a.output/f'{label}.json').write_text(json.dumps(policy,indent=2)+'\n')
(a.output/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
print(json.dumps({k:{m:sum(v['mode']==m for v in p['functions'].values()) for m in ['shift','late','vector','late_vector']} for k,p in policies.items()},indent=2))
