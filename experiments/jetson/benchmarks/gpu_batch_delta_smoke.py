"""Same-buffer GPU codec oracle including uint64 wrap, range escape and SM255/256."""
import ctypes as C,json,struct,sys
from pathlib import Path
import numpy as np
from trace_codec import decode_gpu9
cu=C.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
def bind(name,args):
    f=getattr(cu,name);f.argtypes=args;f.restype=C.c_int;return f
ptr=C.c_uint64;handle=C.c_void_p
init=bind('cuInit',[C.c_uint]);device=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(handle),C.c_int]);current=bind('cuCtxSetCurrent',[handle])
alloc=bind('cuMemAlloc_v2',[C.POINTER(ptr),C.c_size_t]);free=bind('cuMemFree_v2',[ptr]);htod=bind('cuMemcpyHtoD_v2',[ptr,C.c_void_p,C.c_size_t]);dtoh=bind('cuMemcpyDtoH_v2',[C.c_void_p,ptr,C.c_size_t]);load=bind('cuModuleLoad',[C.POINTER(handle),C.c_char_p]);get=bind('cuModuleGetFunction',[C.POINTER(handle),handle,C.c_char_p]);launch=bind('cuLaunchKernel',[handle,*([C.c_uint]*7),handle,C.POINTER(handle),C.POINTER(handle)])
assert init(0)==0;dev=C.c_int();assert device(C.byref(dev),0)==0;ctx=handle();assert retain(C.byref(ctx),dev)==0;assert current(ctx)==0
module=handle();assert load(C.byref(module),sys.argv[1].encode())==0
functions={}
for name in ('neutrino_capture_pack_batch','neutrino_capture_delta_init','neutrino_capture_delta_batch'):
    f=handle();assert get(C.byref(f),module,name.encode())==0;functions[name]=f
def call(name,grid,*args):
    a=(handle*len(args))(*(C.addressof(x) for x in args));assert launch(functions[name],grid,1,1,256,1,1,0,None,a,None)==0
results=[];dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')])
for mode in ('batch13','delta9','delta_wrap','delta_range_fallback','delta_sm_fallback'):
    memories=[];descriptors=[];rows=[];blocks=0
    sizes=[1,33,4096,5000,607744,8192,65]
    for j,n in enumerate(sizes):
        a=np.empty(n,dtype=dtype);a['sm']=np.arange(n)%3;a['sm'][a['sm']==1]=81;a['sm'][a['sm']==2]=255
        a['elapsed']=np.uint32(2**32-1)-np.arange(n,dtype='uint32')
        a['start']=np.uint64(2**64-1)-np.arange(n,dtype='uint64')
        if mode=='delta_wrap':a['start']=np.arange(n,dtype='uint64')-np.uint64(n//2)
        if mode=='delta_range_fallback' and n>1:a['sm']=0;a['start'][::2]=0;a['start'][1::2]=2**40
        if mode=='delta_sm_fallback':a['sm'][-1]=256
        raw=a.tobytes();src=ptr();dst=ptr();sz=n*13 if mode=='batch13' else n*9+2064
        assert alloc(C.byref(src),len(raw))==0 and alloc(C.byref(dst),sz)==0 and htod(src,raw,len(raw))==0
        nb=(n+255)//256;descriptors.append(struct.pack('<QQQII',src.value,dst.value,n,blocks,nb));blocks+=nb
        rows.append((a,dst,sz));memories.extend([src,dst])
    table=b''.join(descriptors);desc=ptr();state=ptr();assert alloc(C.byref(desc),len(table))==0 and htod(desc,table,len(table))==0
    assert alloc(C.byref(state),4 if mode=='batch13' else len(sizes)*1024)==0
    count=C.c_uint(len(sizes));memories.extend([desc,state])
    if mode=='batch13':assert htod(state,b'\0'*4,4)==0;call('neutrino_capture_pack_batch',blocks,desc,count,state)
    else:call('neutrino_capture_delta_init',len(sizes),desc,state);call('neutrino_capture_delta_batch',blocks,desc,count,state)
    flags=[];exact=0
    for a,dst,sz in rows:
        b=C.create_string_buffer(sz);assert dtoh(b,dst,sz)==0;data=b.raw;n=len(a)
        if mode=='batch13':
            recovered=np.empty(n,dtype=dtype);recovered['start']=np.frombuffer(data,dtype='<u8',count=n);recovered['elapsed']=np.frombuffer(data,offset=n*8,dtype='<u4',count=n);recovered['sm']=np.frombuffer(data,offset=n*12,dtype='u1',count=n);decoded=recovered.tobytes();flag=0
        else:
            flag=struct.unpack_from('<I',data)[0];decoded=None if flag else decode_gpu9(data,n)
            expected=1 if mode=='delta_sm_fallback' else 2 if mode=='delta_range_fallback' and n>1 else 0
            assert flag==expected,(mode,n,flag,expected)
        flags.append(flag)
        if decoded is not None:assert decoded==a.tobytes();exact+=n
    for m in memories:assert free(m)==0
    results.append(dict(case=mode,frame_sizes=sizes,exact_records=exact,flags=flags))
Path(sys.argv[2]).write_text(json.dumps(dict(passed=True,cases=results),indent=2)+'\n')
