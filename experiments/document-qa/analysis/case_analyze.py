"""Identical streaming analyzer for raw and exact encoded complete traces."""
import hashlib, json, struct, sys, math, ctypes as C
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/benchmarks'))
from final_trace_queries import frames
import trace_codec
if (ROOT/'code/reference_decode.so').exists():
 lib=C.CDLL(str(ROOT/'code/reference_decode.so'));decoder=lib.case_reference_decode
 decoder.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t];decoder.restype=C.c_int
 def fast_reference(data,n):
  out=np.empty(n*16,dtype=np.uint8)
  assert decoder(data,len(data),out.ctypes.data,n)==0
  return out.tobytes()
 trace_codec.decode_reference=fast_reference
DT=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')])

def sidecar(path,request):
 rows=[]
 for line in path.read_text().splitlines():
  seq,req,token,ok,stream,dims,name=line.split('\t',6)
  if int(req)==request and int(ok):rows.append({'ordinal':int(seq),'token':int(token),'name':name,'dims':list(map(int,dims.split(','))),'stream':stream})
 return rows

def raw_frames(path):
 with Path(path).open('rb') as f:
  first=f.read(8); f.seek(0)
  if first.startswith((b'NTGP',b'NTRF',b'NTFX',b'NTHR',b'NTNV')):
   yield from frames(Path(path));return
  while True:
   off=f.tell();header=f.read(48)
   if not header:break
   assert len(header)==48
   d=struct.unpack('<10IQ',header)
   assert d[7:]==(1,16,32,48)
   n=math.prod(d[:3])*((math.prod(d[3:6])+31)//32)
   data=f.read(n*16);assert len(data)==n*16
   yield off,d,data

def category(name):
 if 'flash_attn' in name and ('combine' in name or 'fixup' in name):return 'attention_aux'
 if 'flash_attn' in name:return 'attention'
 if 'mul_mat' in name:return 'matmul_matvec'
 if 'norm' in name:return 'norm'
 if 'cpy' in name or 'copy' in name:return 'copy'
 return 'other'

def analyze(paths,meta,request,save=None):
 rows=sidecar(meta,request);out=[];digest=hashlib.sha256();records=0
 it=(frame for path in paths for frame in raw_frames(path))
 for row,(_,d,data) in zip(rows,it,strict=True):
  assert row['dims']==list(d[:7]),(row['ordinal'],'identity mismatch')
  a=np.frombuffer(data,dtype=DT);w=(math.prod(d[3:6])+31)//32
  assert np.all(a['sm']<256) and not np.any(a['start']==0xdeadbeefdeadbeef)
  digest.update(struct.pack('<10IQ',*d));digest.update(data);records+=len(a)
  ctas=a.reshape(-1,w);duration=ctas['elapsed'].max(axis=1)
  positions=[.5*(len(duration)-1),.95*(len(duration)-1)]
  indices=sorted(set(int(i) for v in positions for i in (np.floor(v),np.ceil(v))))
  ordered=np.partition(duration,indices)
  med,q95=[float(ordered[int(np.floor(v))])*(1-(v%1))+float(ordered[int(np.ceil(v))])*(v%1) for v in positions]
  val={**row,'category':category(row['name']),'records':len(a),'warps_per_cta':w,
   'median_cta_cycles':med,'p95_cta_cycles':q95,
   'max_cta_cycles':int(duration.max()),'p95_over_median':q95/max(1,med),
   'active_sms':int(np.count_nonzero(np.bincount(a['sm'].astype('int64'),minlength=256)))}
  out.append(val)
  if save and row['category']=='attention':
   np.savez_compressed(Path(save)/f"attention-{row['ordinal']}.npz",records=a,dims=np.array(d[:7]),token=row['token'],name=row['name'])
 totals={}
 for r in out:
  key=('prefill' if r['token']==0 else 'decode')+'/'+r['category']
  t=totals.setdefault(key,{'launches':0,'records':0});t['launches']+=1;t['records']+=r['records']
 return {'request':request,'frames':len(out),'records':records,'decoded_sha256':digest.hexdigest(),
  'totals':totals,'launches':out,'scope':'Every hook-covered launch; unchanged complete warp records; cycles are instrumented observations, not native time'}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--workload',type=Path,required=True);p.add_argument('--sidecar',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 for r in json.loads(a.workload.read_text())['requests']:
  result=analyze(r['trace_paths'],a.sidecar,r['request'],a.out)
  (a.out/f"request-{r['request']}.json").write_text(json.dumps(result,indent=2)+'\n')
