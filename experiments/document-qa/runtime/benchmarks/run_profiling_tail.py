import json,os,shutil,struct,subprocess,sys
from pathlib import Path
import numpy as np
from trace_codec import decode_gpu9,decode_flex
root=Path(__file__).resolve().parents[1];out=root/'results/profiling-tail';out.mkdir(exist_ok=False)
base={'NEUTRINO_ARENA':'1','NEUTRINO_CAPTURE_CLEAR':'1','NEUTRINO_CAPTURE_ENCODER':str(root/'benchmarks/capture_pack.cubin'),'NEUTRINO_ARENA_HOST_MIB':'32','NEUTRINO_ARENA_DEVICE_MIB':'16','NEUTRINO_ARENA_INCREMENTAL':'1','NEUTRINO_ARENA_BATCH_ENCODER':'1','NEUTRINO_ARENA_DECOUPLE':'0','NEUTRINO_ARENA_DELTA':'1','NEUTRINO_ARENA_WRITER':'1'}
spec={'N':None,'S':{},'DW':base,'H2':{**base,'NEUTRINO_ARENA_DELTA_WIDTH':'3','NEUTRINO_ARENA_ELAPSED_WIDTH':'2','NEUTRINO_ARENA_HIER':'1'}}
dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')]);results=[]
def metrics(raw):
 a=np.frombuffer(raw,dtype=dtype);assert len(a)==4096*4 and np.all(a['sm']<82)
 score=a['elapsed'].reshape(4096,4).max(axis=1).astype('float64');labels=np.arange(4096)%8==0
 order=np.argsort(score);ranks=np.empty(4096,dtype='float64');values,starts,counts=np.unique(score[order],return_index=True,return_counts=True)
 for start,count in zip(starts,counts):ranks[order[start:start+count]]=start+(count+1)/2
 pos=labels.sum();auc=(ranks[labels].sum()-pos*(pos+1)/2)/(pos*(4096-pos));top=np.argsort(score)[-pos:]
 return dict(auc=float(auc),top_k_recall=float(labels[top].mean()),tail_median_cycles=float(np.median(score[labels])),ordinary_median_cycles=float(np.median(score[~labels])),max_elapsed=int(a['elapsed'].max()))
for r in range(3):
 order=list(spec);order=order[r:]+order[:r]
 for name in order:
  dest=out/f'r{r+1}-{name}';dest.mkdir();trace=Path('/dev/shm/neutrino-mechanism-20261004')/f'tail-{r}-{name}';trace.mkdir(exist_ok=False)
  env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')};env.update(NEUTRINO_BENCH_DIR=str(dest),PATH='/usr/local/cuda/bin:'+env.get('PATH',''))
  cmd=[sys.executable,str(root/'benchmarks/profiling_tail_workload.py')]
  if name!='N':
   env.update(spec[name]);cmd=[sys.executable,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null','--kernel','neutrino_controlled_tail','--ablation-mode','files','--tracedir',str(trace),*cmd]
  try:
   with (dest/'run.log').open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=log,check=True,timeout=180)
   w=json.loads((dest/'workload.json').read_text());frames=[]
   for p in sorted(trace.glob('*/result/*')):
    if p.suffix not in ['.bin','.nts']:continue
    with p.open('rb') as f:
     magic=f.read(8) if p.suffix=='.nts' else None
     while True:
      h=f.read(48)
      if not h:break
      gx,gy,gz,bx,by,bz,_,maps,item,div,offset=struct.unpack('<10IQ',h);assert maps==1 and item==16 and div==32 and offset==48;n=gx*gy*gz*((bx*by*bz+31)//32)
      tag=0;packed=0
      if magic:packed,=struct.unpack('<Q',f.read(8));tag=packed>>62
      data=f.read((packed&((1<<62)-1)) or n*16)
      if tag==2:data=decode_gpu9(data,n)
      elif tag==3:data=decode_flex(data,n)
      frames.append(metrics(data))
   assert len(frames)==(0 if name=='N' else 7)
   row=dict(round=r+1,condition=name,workload=w,profiles=frames[2:])
   (dest/'result.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
  finally:shutil.rmtree(trace)
assert len({x['workload']['digest'] for x in results})==1
(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
