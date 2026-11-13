"""Build/validate, then run finite serial matrices on the dedicated GPU."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from arena_checks import sequence
from trace_codec import check_gpu
root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True)
py=sys.executable;env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
def call(cmd,**kw):subprocess.run(cmd,cwd=root,env=env,check=True,**kw)
call([py,'neutrino/build.py'])
cuda='/usr/local/cuda/targets/x86_64-linux'
call(['cc','-shared','-fPIC','-O3','benchmarks/profile_preload.c','-ldl','-o','neutrino/build/profile-preload.so'])
call(['cc','-shared','-fPIC','-O3','benchmarks/deep_cupti_gpuonly.c','-I'+cuda+'/include','-L'+cuda+'/lib','-Wl,-rpath,'+cuda+'/lib','-lcupti','-ldl','-lpthread','-o','benchmarks/deep_cupti_gpuonly.so'])
call([py,'-m','unittest','discover','-s','tests','-p','test_arena_ring.py'])
cases={'coupled':{'NEUTRINO_ARENA_DECOUPLE':'0'},'decoupled':{},'template':{'NEUTRINO_ARENA_TEMPLATE':'1'},'frame':{'NEUTRINO_ARENA_FRAMES':'1'}}
checks=[]
for name,extra in cases.items():
    dest=out/'stateful-validation'/name;dest.mkdir(parents=True,exist_ok=False)
    trace=Path('/dev/shm/neutrino-arena-20261003')/('stateful-'+name);trace.mkdir(parents=True,exist_ok=False)
    e=dict(env,NEUTRINO_BENCH_DIR=str(dest),NEUTRINO_ARENA='1',NEUTRINO_ARENA_DIAG='1',NEUTRINO_ARENA_VERIFY='1',NEUTRINO_CAPTURE_CLEAR='2',NEUTRINO_ARENA_HOST_MIB='32',NEUTRINO_ARENA_DEVICE_MIB='16',NEUTRINO_CAPTURE_ENCODER=str(root/'benchmarks/capture_pack.cubin'),**extra)
    cmd=[py,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null','--kernel','neutrino_arena_add','--ablation-mode','files','--tracedir',str(trace),py,str(root/'benchmarks/arena_gpu_validation.py')]
    try:
        with (dest/'log.txt').open('w') as log:subprocess.run(cmd,cwd=root,env=e,stdout=log,stderr=log,check=True,timeout=180)
        w=json.loads((dest/'validation.json').read_text());a=json.loads((dest/'capture-summary.json').read_text())['arena_engines'][0]
        paths=list(trace.glob('*/result/*.nts'));assert len(paths)==1
        s=sequence(paths[0]);r=check_gpu(paths[0],True)
        assert s['frames']==w['launches']==a['calls'] and r['zero_records']==0
        assert a['outstanding_bytes']==0 and a['published_total']==a['consumed_total']==a['retired_total']
        if name=='template':assert a['template_invalidations']>=2 and a['template_hits']>0 and a['template_builds']>=2
        checks.append(dict(case=name,**w,trace=s,records=r,summary=a));(dest/'checks.json').write_text(json.dumps(checks[-1],indent=2)+'\n')
    finally:shutil.rmtree(trace)
(out/'stateful-validation/summary.json').write_text(json.dumps(checks,indent=2)+'\n')
for stage in ['final-smoke','main','detail','observer','long','context','gpu','regression']:
    print('SUITE STAGE',stage,flush=True);call([py,'benchmarks/run_arena_experiment.py','--stage',stage])
(out/'suite-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
