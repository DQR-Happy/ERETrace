"""Check generic tail trimming/growth against a stateful CPU oracle."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from arena_checks import sequence
from trace_codec import check_gpu
root=Path(__file__).resolve().parents[1];out=root/'results/mechanism-stateful'
checks=[]
base={'NEUTRINO_ARENA_INCREMENTAL':'1','NEUTRINO_ARENA_BATCH_ENCODER':'1','NEUTRINO_ARENA_DELTA':'1','NEUTRINO_ARENA_DECOUPLE':'0','NEUTRINO_ARENA_WRITER':'1','NEUTRINO_ARENA_DELTA_WIDTH':'3','NEUTRINO_ARENA_ELAPSED_WIDTH':'2'}
cases={'flex32':base,'hier2':{**base,'NEUTRINO_ARENA_HIER':'1'},'hier2_checked':{**base,'NEUTRINO_ARENA_HIER':'1','NEUTRINO_ARENA_CHECK_FIRST':'1','NEUTRINO_ARENA_RELEASE_CHECKED':'1'},'hier2_escape':{**base,'NEUTRINO_ARENA_HIER':'1','NEUTRINO_ARENA_FORCE_FALLBACK':'1'}}
for name,extra in cases.items():
    dest=out/name;dest.mkdir(parents=True,exist_ok=False)
    trace=Path('/dev/shm/neutrino-mechanism-20261004')/('stateful-'+name);trace.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),NEUTRINO_BENCH_DIR=str(dest),NEUTRINO_ARENA='1',NEUTRINO_ARENA_DIAG='1',NEUTRINO_ARENA_VERIFY='1',NEUTRINO_CAPTURE_CLEAR='2',NEUTRINO_ARENA_HOST_MIB='32',NEUTRINO_ARENA_DEVICE_MIB='16',NEUTRINO_CAPTURE_ENCODER=str(root/'benchmarks/capture_pack.cubin'),**extra)
    cmd=[sys.executable,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null','--kernel','neutrino_arena_add','--ablation-mode','files','--tracedir',str(trace),sys.executable,str(root/'benchmarks/arena_gpu_validation.py')]
    try:
        with (dest/'log.txt').open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=log,check=True,timeout=180)
        w=json.loads((dest/'validation.json').read_text());a=json.loads((dest/'capture-summary.json').read_text())['arena_engines'][0]
        paths=list(trace.glob('*/result/*.nts'));assert len(paths)==1
        s=sequence(paths[0]);r=check_gpu(paths[0],True)
        assert s['frames']==w['launches']==a['calls'] and r['zero_records']==0
        assert a['outstanding_bytes']==0 and a['published_total']==a['consumed_total']==a['retired_total']
        checks.append(dict(case=name,**w,trace=s,records=r,summary=a));(dest/'checks.json').write_text(json.dumps(checks[-1],indent=2)+'\n')
    finally:shutil.rmtree(trace)
(out/'summary.json').write_text(json.dumps(checks,indent=2)+'\n')
