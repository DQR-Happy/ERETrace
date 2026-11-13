"""Consume one trace locally on the GPU host; retain counts, never raw records."""
import json,os,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from trace_codec import decode_gpu9,decode_flex
path=Path(sys.argv[1]);out=Path(sys.argv[2]);root=Path(__file__).resolve().parents[1]
ids=[list(map(int,x.split(','))) for x in path.with_suffix('.ids').read_text().splitlines()]
selected=[];range_rows=[];ranges_only=os.environ.get('MECHANISM_RANGES_ONLY')=='1';hier_micro=os.environ.get('MECHANISM_HIER_MICRO')=='1'
with path.open('rb') as f:
 magic=f.read(8)
 for launch,token,offset,raw,size in ids:
  h=f.read(48);packed,=struct.unpack('<Q',f.read(8));tag=packed>>62;size=(packed&((1<<62)-1)) or raw;data=f.read(size)
  if raw<65536 or (not ranges_only and token!=1):continue
  n=raw//16
  if tag==2:data=decode_gpu9(data,n)
  elif tag==3:data=decode_flex(data,n)
  elif tag==0 and packed:
   a=np.empty(n,dtype=[('start','<u8'),('elapsed','<u4'),('sm','<u4')]);a['start']=np.frombuffer(data,offset=0,count=n,dtype='<u8');a['elapsed']=np.frombuffer(data,offset=8*n,count=n,dtype='<u4');a['sm']=np.frombuffer(data,offset=12*n,count=n,dtype='u1');data=a.tobytes()
  assert len(data)==raw

  a=np.frombuffer(data,dtype=[('start','<u8'),('elapsed','<u4'),('sm','<u4')]);span=0
  for sm in np.unique(a['sm']):
   values=a['start'][a['sm']==sm];d=(values-values[0]).view('int64');span=max(span,int(d.max())-int(d.min()))
  dims=struct.unpack('<8I',h[:32]);wp=(dims[3]*dims[4]*dims[5]+31)//32;starts=a['start'].reshape(-1,wp);sms=a['sm'].reshape(-1,wp);local=(starts-starts[:,0,None]).view('int64')
  if not ranges_only:selected.append((data,wp))
  range_rows.append(dict(launch=launch,token=token,n=n,warps_per_cta=wp,raw_bytes=raw,elapsed_max=int(a['elapsed'].max()),same_sm_span=span,sm_count=len(np.unique(a['sm'])),cta_local_min=int(local.min()),cta_local_max=int(local.max()),cta_sm_consistent=bool(np.all(sms==sms[:,0,None]))))
assert range_rows
out.mkdir(exist_ok=True)
if ranges_only:
 (out/'range-summary.json').write_text(json.dumps(dict(source_frame_count=len(ids),scope='all encoded frames of one request',frames=range_rows),indent=2)+'\n')
 (out/'complete.json').write_text(json.dumps(dict(ranges_only=True,raw_input_deleted=True,frames=len(range_rows)))+'\n');sys.exit(0)
with tempfile.TemporaryDirectory(prefix='neutrino-codec-control-',dir='/dev/shm') as temp:
 p=Path(temp)/'input.bin'
 with p.open('wb') as f:
  f.write(struct.pack('<I',len(selected)))
  for b,wp in selected:
   f.write(struct.pack('<Q',len(b)//16))
   if hier_micro:f.write(struct.pack('<I',wp))
   f.write(b)
 for r in range(3):
  with (out/f'process-{r+1}.json').open('w') as f:
   subprocess.run([str(root/('benchmarks/hier_service_micro' if hier_micro else 'benchmarks/codec_service_micro')),str(p),temp,str(r)],stdout=f,check=True)
(out/'range-summary.json').write_text(json.dumps(dict(source_frame_count=len(ids),token=1,frames=range_rows),indent=2)+'\n')
(out/'complete.json').write_text(json.dumps(dict(processes=3,raw_input_deleted=True,frames=len(selected)))+'\n')
