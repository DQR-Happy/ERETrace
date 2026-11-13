"""Repair only four incomplete variable-size nvCOMP copy diagnostic windows."""
import copy,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/neutrino-b16-attribution-20261007')
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
state=read(ROOT/'gpu-status.json');state['complete']=False;save(ROOT/'gpu-status.json',state)
cuda='/usr/local/cuda/targets/x86_64-linux'
lib=ROOT/'build/deep_copy8192.so'
subprocess.run(['cc','-O3','-shared','-fPIC',str(ROOT/'source/deep_cupti_copy8192.c'),'-I'+cuda+'/include','-L'+cuda+'/lib','-Wl,-rpath,'+cuda+'/lib','-lcupti','-ldl','-lpthread','-o',str(lib)],check=True)
env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')}
env['PATH']='/root/miniconda3/bin:/usr/local/cuda/bin:'+env.get('PATH','')
env['PYTHONPATH']=str(ROOT/'source')+':'+str(ROOT/'collector-old/benchmarks')+':'+str(ROOT/'collector-old')
replaced=[]
for batch,rounds in [(4,1),(8,1),(16,2)]:
    name=f'b{batch}-gpu';old=ROOT/'gpu'/name
    cfg=read(old/'config.json');spec=copy.deepcopy(cfg['specs']['NC'])
    spec['env']['DEEP_CUPTI_LIBRARY']=str(lib)
    sp=ROOT/f'b{batch}-gpu-repair-specs.json';save(sp,{'NC':spec})
    out=ROOT/'gpu-repair'/name
    cmd=[sys.executable,str(ROOT/'source/run_arena_matrix.py'),'--output',str(out),'--trace-root','/dev/shm/neutrino-b16-gpu-repair-20261007','--spec',str(sp),'--conditions','NC','--rounds',str(rounds),'--seed','21042','--timeout','1200','--',*cfg['command']]
    print(name,'REPAIR START',flush=True);subprocess.run(cmd,env=env,check=True)
    nr=read(out/'runs.json');ors=read(old/'runs.json');byround={r['round']:r for r in nr}
    for r in nr:
        rel=f"r{r['round']}-NC";d=old/rel;x=read(d/'cupti.json')
        assert x['device_copy_group_overflow'] and not x['dropped'] and not x['overflow']
        z=read(out/rel/'cupti.json');assert not z['device_copy_group_overflow'] and not z['dropped'] and not z['overflow']
        assert r['output_digest']==next(v for v in ors if v['condition']=='NC' and v['round']==r['round'])['output_digest']
        destination=ROOT/'gpu-incomplete'/name/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(d),str(destination));shutil.move(str(out/rel),str(d))
        replaced.append(dict(matrix=name,round=r['round'],previous_overflow=x['device_copy_group_overflow'],new_copy_groups=len(z['device_copy_groups'])))
    save(old/'initial-runs.json',ors)
    save(old/'runs.json',[byround[r['round']] if r['condition']=='NC' else r for r in ors])
    cfg['diagnostic_repair']=dict(conditions=['NC'],capacity=8192,library=str(lib));save(old/'config.json',cfg)
    print(name,'REPAIR COMPLETE',flush=True)
tmp=Path('/dev/shm/neutrino-b16-gpu-repair-20261007');assert not list(tmp.iterdir());tmp.rmdir()
save(ROOT/'gpu-copy-repair.json',dict(complete=True,windows=replaced,tmpfs_absent=True,library=str(lib),binary_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ROOT/'source/deep_cupti_copy8192.c').read_bytes()).hexdigest()))
state['complete']=True;state['copy_capacity_repair']='gpu-copy-repair.json';save(ROOT/'gpu-status.json',state)
print('COPY REPAIR COMPLETE',flush=True)
