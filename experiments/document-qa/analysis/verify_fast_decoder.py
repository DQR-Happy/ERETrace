"""Compare every encoded frame to the independent NumPy decoder byte-for-byte."""
from pathlib import Path
import json,sys,struct,ctypes as C,hashlib
import zstandard as z
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'source/benchmarks'))
from trace_codec import decode_reference
lib=C.CDLL(str(r/'code/reference_decode.so'));fn=lib.case_reference_decode
fn.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t];fn.restype=C.c_int
files=list((r/'capture/r0-ispa_ere-validate').glob('trace*.zst'));assert len(files)==1
checked=records=frames=0;sha=hashlib.sha256()
with files[0].open('rb') as raw,z.ZstdDecompressor().stream_reader(raw) as f:
 magic=f.read(8);assert magic[:4]==b'NTRF'
 while True:
  header=f.read(48)
  if not header:break
  assert len(header)==48;d=struct.unpack('<10IQ',header);n=int(np.prod(d[:3]))*((int(np.prod(d[3:6]))+31)//32)
  v,=struct.unpack('<Q',f.read(8));tag=v>>62;size=v&((1<<62)-1);data=f.read(size or n*16);assert len(data)==(size or n*16)
  if tag==3:
   expected=decode_reference(data,n);out=np.empty(n*16,dtype='u1');assert fn(data,len(data),out.ctypes.data,n)==0
   assert out.tobytes()==expected;checked+=1;records+=n;sha.update(expected)
  frames+=1
j={'passed':True,'checked_encoded_frames':checked,'checked_records':records,'all_frames':frames,'decoded_encoded_bytes_sha256':sha.hexdigest(),'comparison':'C++ optimized decoder vs independent NumPy decoder: exact bytes, every encoded frame; capture itself independently checked against original GPU Map'}
(r/'fast-decoder-validation.json').write_text(json.dumps(j,indent=2)+'\n');print(j)
