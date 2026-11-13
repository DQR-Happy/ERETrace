"""Direct CUDA oracle: exact 64-bit edge values and an explicit overflow flag."""
import ctypes as C
import json
from pathlib import Path
import struct
import sys

cu=C.CDLL(__import__('os').environ.get('CUDA_DRIVER_LIBRARY', 'libcuda.so.1'))
def bind(name,args):
    f=getattr(cu,name);f.argtypes=args;f.restype=C.c_int;return f
ptr=C.c_uint64;handle=C.c_void_p
init=bind('cuInit',[C.c_uint]);device=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int])
retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(handle),C.c_int]);current=bind('cuCtxSetCurrent',[handle])
alloc=bind('cuMemAlloc_v2',[C.POINTER(ptr),C.c_size_t]);free=bind('cuMemFree_v2',[ptr])
htod=bind('cuMemcpyHtoD_v2',[ptr,C.c_void_p,C.c_size_t]);dtoh=bind('cuMemcpyDtoH_v2',[C.c_void_p,ptr,C.c_size_t])
load=bind('cuModuleLoad',[C.POINTER(handle),C.c_char_p]);get=bind('cuModuleGetFunction',[C.POINTER(handle),handle,C.c_char_p])
launch=bind('cuLaunchKernel',[handle,*([C.c_uint]*7),handle,C.POINTER(handle),C.POINTER(handle)])
assert init(0)==0;dev=C.c_int();assert device(C.byref(dev),0)==0;ctx=handle();assert retain(C.byref(ctx),dev)==0;assert current(ctx)==0
module=handle();assert load(C.byref(module),sys.argv[1].encode())==0;fun=handle();assert get(C.byref(fun),module,b'neutrino_capture_pack')==0
results=[]
for bad in (False,True):
    rows=[(2**64-1-i,2**32-1-i,[0,81,255][i%3]) for i in range(5000)]
    if bad:rows[100]=(123,456,256)
    raw=b''.join(struct.pack('<QII',*r) for r in rows);src=ptr();dst=ptr();err=ptr()
    assert alloc(C.byref(src),len(raw))==0 and alloc(C.byref(dst),len(rows)*13)==0 and alloc(C.byref(err),4)==0
    assert htod(src,raw,len(raw))==0 and htod(err,b'\0'*4,4)==0
    n=C.c_uint64(len(rows));args=(handle*4)(C.addressof(src),C.addressof(dst),C.addressof(n),C.addressof(err))
    assert launch(fun,(len(rows)+255)//256,1,1,256,1,1,0,None,args,None)==0
    out=C.create_string_buffer(len(rows)*13);flag=C.c_uint32();assert dtoh(out,dst,len(out))==0 and dtoh(C.byref(flag),err,4)==0
    assert flag.value==int(bad)
    if not bad:
        data=out.raw;n=len(rows)
        decoded=[(struct.unpack_from('<Q',data,i*8)[0],struct.unpack_from('<I',data,n*8+i*4)[0],data[n*12+i]) for i in range(n)]
        assert decoded==rows
    for p in (src,dst,err):assert free(p)==0
    results.append(dict(records=len(rows),invalid_sm=bad,overflow_flag=flag.value,exact_roundtrip=not bad))
Path(sys.argv[2]).parent.mkdir(parents=True,exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps(dict(passed=True,cases=results),indent=2)+'\n')
