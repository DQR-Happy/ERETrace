"""Exact GPU field-width family: independent same-buffer oracle and escape cases."""
import ctypes as C,json,struct,sys
from pathlib import Path
import numpy as np
from trace_codec import decode_flex
cu=C.CDLL(__import__('os').environ.get('CUDA_DRIVER_LIBRARY', 'libcuda.so.1'))
def bind(n,a):
 f=getattr(cu,n);f.argtypes=a;f.restype=C.c_int;return f
ptr=C.c_uint64;handle=C.c_void_p
init=bind('cuInit',[C.c_uint]);device=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(handle),C.c_int]);current=bind('cuCtxSetCurrent',[handle])
alloc=bind('cuMemAlloc_v2',[C.POINTER(ptr),C.c_size_t]);free=bind('cuMemFree_v2',[ptr]);htod=bind('cuMemcpyHtoD_v2',[ptr,C.c_void_p,C.c_size_t]);dtoh=bind('cuMemcpyDtoH_v2',[C.c_void_p,ptr,C.c_size_t]);load=bind('cuModuleLoad',[C.POINTER(handle),C.c_char_p]);get=bind('cuModuleGetFunction',[C.POINTER(handle),handle,C.c_char_p]);launch=bind('cuLaunchKernel',[handle,*([C.c_uint]*7),handle,C.POINTER(handle),C.POINTER(handle)])
assert init(0)==0;dev=C.c_int();assert device(C.byref(dev),0)==0;ctx=handle();assert retain(C.byref(ctx),dev)==0;assert current(ctx)==0
module=handle();assert load(C.byref(module),sys.argv[1].encode())==0
func={}
for n in ['neutrino_capture_flex_init','neutrino_capture_flex_batch']:
 f=handle();assert get(C.byref(f),module,n.encode())==0;func[n]=f
def call(n,grid,*args):
 a=(handle*len(args))(*(C.addressof(x) for x in args));assert launch(func[n],grid,1,1,256,1,1,0,None,a,None)==0
results=[];dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')])
for dw in [2,3,4]:
 for ew in [2,3,4]:
  for mode in ['exact_wrap','delta_escape','elapsed_escape','sm_escape']:
   rows=[];desc=[];mem=[];blocks=0
   for j,n in enumerate([1,33,4096,8193,607744]):
    a=np.empty(n,dtype=dtype);idx=np.arange(n,dtype='uint64');a['sm']=idx%3;a['sm'][a['sm']==1]=81;a['sm'][a['sm']==2]=255
    a['start']=(idx%1024)-np.uint64(512);a['elapsed']=np.uint32((1<<(ew*8))-1)-(idx%1024).astype('uint32')
    expected=0
    if mode=='delta_escape' and n>1:a['sm']=0;a['start'][::2]=0;a['start'][1::2]=1<<(dw*8+1);expected=2
    if mode=='elapsed_escape' and ew<4:a['elapsed'][-1]=1<<(ew*8);expected=4
    if mode=='sm_escape':a['sm'][-1]=256;expected=1
    raw=a.tobytes();src=ptr();dst=ptr();sz=2064+n*(dw+ew+1)
    assert alloc(C.byref(src),len(raw))==0 and alloc(C.byref(dst),sz)==0 and htod(src,raw,len(raw))==0
    nb=(n+255)//256;desc.append(struct.pack('<QQQII',src.value,dst.value,n,blocks,nb));blocks+=nb;mem += [src,dst];rows.append((raw,dst,sz,n,expected))
   table=b''.join(desc);d=ptr();state=ptr();assert alloc(C.byref(d),len(table))==0 and htod(d,table,len(table))==0 and alloc(C.byref(state),len(rows)*1024)==0;mem += [d,state]
   call('neutrino_capture_flex_init',len(rows),d,state,C.c_uint(dw),C.c_uint(ew));call('neutrino_capture_flex_batch',blocks,d,C.c_uint(len(rows)),state,C.c_uint(dw),C.c_uint(ew))
   exact=0;flags=[]
   for raw,dst,sz,n,expected in rows:
    b=C.create_string_buffer(sz);assert dtoh(b,dst,sz)==0;flag=struct.unpack_from('<I',b.raw)[0];assert flag==expected,(dw,ew,mode,n,flag,expected)
    if not flag:assert decode_flex(b.raw,n)==raw;exact+=n
    flags.append(flag)
   for p in mem:assert free(p)==0
   results.append(dict(dw=dw,ew=ew,case=mode,flags=flags,exact_records=exact))
Path(sys.argv[2]).write_text(json.dumps(dict(passed=True,cases=results),indent=2)+'\n')
