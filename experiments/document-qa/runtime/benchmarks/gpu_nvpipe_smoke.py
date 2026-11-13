import ctypes as C,json,sys
from pathlib import Path
import numpy as np
cu=C.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
def bind(n,args):
 f=getattr(cu,n);f.argtypes=args;f.restype=C.c_int;return f
P=C.c_uint64;H=C.c_void_p
init=bind('cuInit',[C.c_uint]);devget=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(H),C.c_int]);current=bind('cuCtxSetCurrent',[H]);alloc=bind('cuMemAlloc_v2',[C.POINTER(P),C.c_size_t]);free=bind('cuMemFree_v2',[P]);htod=bind('cuMemcpyHtoD_v2',[P,C.c_void_p,C.c_size_t]);dtoh=bind('cuMemcpyDtoH_v2',[C.c_void_p,P,C.c_size_t]);sync=bind('cuCtxSynchronize',[])
assert init(0)==0;dev=C.c_int();assert devget(C.byref(dev),0)==0;ctx=H();assert retain(C.byref(ctx),dev)==0 and current(ctx)==0
lib=C.CDLL(sys.argv[1]);lib.nvpipe_create.argtypes=[C.c_size_t,C.c_uint,C.c_uint,C.c_uint];lib.nvpipe_create.restype=H
lib.nvpipe_bound.argtypes=[H,C.c_size_t,C.c_size_t];lib.nvpipe_bound.restype=C.c_size_t
class Frame(C.Structure):_fields_=[('src',C.c_uint64),('bytes',C.c_uint64)]
lib.nvpipe_submit.argtypes=[H,C.c_uint,C.POINTER(Frame),C.c_uint,C.c_size_t,C.c_uint64,H];lib.nvpipe_submit.restype=C.POINTER(C.c_uint64)
lib.nvpipe_decode.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_uint,C.c_uint];lib.nvpipe_decode.restype=C.c_int
lib.nvpipe_destroy.argtypes=[H];lib.nvpipe_space.argtypes=[H,C.POINTER(C.c_uint64)]
results=[];rng=np.random.default_rng(21042)
for mode in [1,2]:
 for chunk in [65536,262144]:
  state=lib.nvpipe_create(16<<20,64,chunk,mode);rows=[];mem=[];frames=[];total=0
  for size in [16,512,16368,16384,65536,65552,262144,9723904]:
   a=rng.integers(0,2**64,size=size//8,dtype='uint64');raw=a.tobytes();src=P();assert alloc(C.byref(src),size)==0 and htod(src,raw,size)==0;mem.append(src);frames.append(Frame(src.value,size));rows.append(raw);total+=lib.nvpipe_bound(state,size,16384)
  out=P();assert alloc(C.byref(out),total)==0;mem.append(out);meta=lib.nvpipe_submit(state,0,(Frame*len(frames))(*frames),len(frames),16384,out.value,None);assert sync()==0;assert meta[0]<=total
  b=C.create_string_buffer(meta[0]);assert dtoh(b,out,meta[0])==0;offset=0
  for i,raw in enumerate(rows):
   n=meta[i+1];payload=b.raw[offset:offset+n];offset+=n
   if len(raw)>=16384:
    dst=C.create_string_buffer(len(raw));assert lib.nvpipe_decode(payload,len(payload),dst,len(raw),mode,chunk)==0;assert dst.raw==raw
   else:assert payload==raw
  assert offset==meta[0];space=(C.c_uint64*5)();lib.nvpipe_space(state,space);results.append(dict(mode=mode,chunk=chunk,frames=len(rows),raw_bytes=sum(map(len,rows)),encoded_bytes=offset,workspace=list(space),exact=True))
  lib.nvpipe_destroy(state)
  for p in mem:assert free(p)==0
Path(sys.argv[2]).write_text(json.dumps(dict(passed=True,cases=results),indent=2)+'\n')
