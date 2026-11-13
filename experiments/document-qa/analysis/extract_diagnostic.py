"""Retain the three full DMAT files and exact arrays from a complete request."""
import json,struct,hashlib,shutil,os
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]; diag=root/os.environ.get('CASE_DIAGNOSTIC_DIR','diagnostic-v7');out=diag/'extracted';out.mkdir(exist_ok=True)
work=json.loads((diag/'workload.json').read_text())['requests'][0]
meta=[]
for line in (diag/'launches.tsv').read_text().splitlines():
 seq,req,token,ok,stream,dims,name=line.split('\t',6)
 if int(req)==0 and int(ok):meta.append(dict(ordinal=int(seq),token=int(token),name=name,dims=list(map(int,dims.split(',')))))
files=sorted(map(Path,work['trace_paths']),key=lambda p:float(p.stem));assert len(files)==len(meta)
parameters={int(row.split('\t')[0]):[int(x,16) for x in row.split('\t')[3:]] for row in (diag/'launches.tsv.params.tsv').read_text().splitlines()}
result=[];total_records=0;digest=hashlib.sha256()
for path,row in zip(files,meta,strict=True):
 with path.open('rb') as f:
  head=f.read(32);d=struct.unpack('<8I',head);assert list(d[:7])==row['dims']
  sections=[struct.unpack('<IIQ',f.read(16)) for _ in range(d[7])]
  nwarp=int(np.prod(d[:3]))*((int(np.prod(d[3:6]))+31)//32)
  assert sections[0][0:2]==(16,32)
  f.seek(sections[0][2]);base=f.read(nwarp*16);assert len(base)==nwarp*16
  digest.update(head);digest.update(base);total_records+=nwarp
  if d[7]==1:continue
  assert d[7]==2 and row['ordinal'] in parameters
  size,div,offset=sections[1];assert div==1 and size%16==0 and size>0
  threads=int(np.prod(d[:6]));f.seek(offset);payload=f.read(threads*size);assert len(payload)==threads*size
  a=np.frombuffer(payload,dtype=[('clock','<u8'),('addr','<u8')]).reshape(threads,size//16)
  target=out/f"dmat-token{row['token']}.bin";shutil.copyfile(path,target)
  np.savez_compressed(out/f"dmat-token{row['token']}.npz",dmat=a,base=np.frombuffer(base,dtype=[('start','<u8'),('elapsed','<u4'),('sm','<u4')]),dims=np.array(d[:7]),pointers=np.array(parameters[row['ordinal']],dtype=np.uint64))
  valid=a['clock']!=0;memory=valid&(a['addr']!=0);branch=valid&(a['addr']==0)
  assert np.all((a['clock'][:,1:]>=a['clock'][:,:-1]) | (a['clock'][:,1:]==0))
  result.append({**row,'slots_per_thread':size//16,'threads':threads,'valid_memory_events':int(memory.sum()),'branch_events':int(branch.sum()),'padding_slots':int((~valid).sum()),'raw_file':str(target),'raw_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'pointers':parameters[row['ordinal']]})
assert len(result)==3 and {r['token'] for r in result}=={0,1,63}
summary={'complete_base_frames':len(files),'complete_base_records':total_records,'complete_base_sha256':digest.hexdigest(),'detail':result,'semantics':'All base warp records retained across entire inference; all memory/branch events of three declared natural attention calls retained; no event sampling'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
