"""Legal CUDA CTA sizes outside the 4/8-warp LLM subset; no kernel replay."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from trace_codec import check_gpu
from arena_checks import sequence
root=Path(__file__).resolve().parents[1];out=root/'results/reference-shapes';out.mkdir(exist_ok=False);rows=[]
base=dict(NEUTRINO_ARENA='1',NEUTRINO_CAPTURE_CLEAR='1',NEUTRINO_CAPTURE_ENCODER=str(root/'benchmarks/capture_pack.cubin'),NEUTRINO_ARENA_HOST_MIB='32',NEUTRINO_ARENA_DEVICE_MIB='16',NEUTRINO_ARENA_INCREMENTAL='1',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',NEUTRINO_ARENA_DELTA='1',NEUTRINO_ARENA_WRITER='1',NEUTRINO_ARENA_DELTA_WIDTH='3',NEUTRINO_ARENA_ELAPSED_WIDTH='2',NEUTRINO_ARENA_HIER='1',NEUTRINO_ARENA_REFERENCE='1',NEUTRINO_ARENA_VERIFY='1')
conditions={'R0':base,'T4':{**base,'NEUTRINO_ARENA_TILE':'4'},'BP':{**base,'NEUTRINO_ARENA_LOCAL_BITS':'12'},'T4P':{**base,'NEUTRINO_ARENA_TILE':'4','NEUTRINO_ARENA_LOCAL_BITS':'12'}}
for threads in [96,128,160,224,256]:
 for name,vars in conditions.items():
  dest=out/f'b{threads}-{name}';dest.mkdir();trace=Path('/dev/shm/neutrino-reference-20261004')/dest.name;trace.mkdir(exist_ok=False)
  env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')};env.update(vars,PATH='/usr/local/cuda/bin:'+env.get('PATH',''))
  cmd=[sys.executable,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null','--kernel','neutrino_reference_shape','--ablation-mode','files','--tracedir',str(trace),str(root/'benchmarks/reference_shape_workload'),str(threads),str(dest)]
  try:
   with (dest/'run.log').open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=log,check=True,timeout=180)
   w=json.loads((dest/'workload.json').read_text());a=json.loads((dest/'capture-summary.json').read_text())['arena_engines'][0];paths=list(trace.glob('*/result/*.nts'));assert len(paths)==7
   structs=[sequence(p) for p in paths];valid=[check_gpu(p,True) for p in paths];assert a['calls']==7 and a['outstanding_bytes']==0
   assert all(x['frames']==1 and x['nonzero_records']==4096*((threads+31)//32) and not x['zero_records'] for x in valid)
   row=dict(condition=name,**w,summary=a,structures=structs,records=valid);(dest/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
  finally:shutil.rmtree(trace)
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
