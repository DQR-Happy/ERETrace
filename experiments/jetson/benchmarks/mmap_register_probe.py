"""Feasibility only: completed device data -> registered tmpfs mapping. No model."""
import ctypes as C,json,mmap,os,statistics,time,sys
from pathlib import Path
cu=C.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
def bind(n,args):
    f=getattr(cu,n);f.argtypes=args;f.restype=C.c_int;return f
H=C.c_void_p;P=C.c_uint64
init=bind('cuInit',[C.c_uint]);devget=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(H),C.c_int]);setctx=bind('cuCtxSetCurrent',[H]);alloc=bind('cuMemAlloc_v2',[C.POINTER(P),C.c_size_t]);free=bind('cuMemFree_v2',[P]);fill=bind('cuMemsetD8_v2',[P,C.c_ubyte,C.c_size_t]);sync=bind('cuCtxSynchronize',[]);register=bind('cuMemHostRegister_v2',[H,C.c_size_t,C.c_uint]);unregister=bind('cuMemHostUnregister',[H]);dtoh=bind('cuMemcpyDtoH_v2',[H,P,C.c_size_t]);errname=bind('cuGetErrorName',[C.c_int,C.POINTER(C.c_char_p)])
assert init(0)==0;dev=C.c_int();assert devget(C.byref(dev),0)==0;ctx=H();assert retain(C.byref(ctx),dev)==0 and setctx(ctx)==0
out=Path(sys.argv[1]);temp=Path('/dev/shm/neutrino-register-20261004');temp.mkdir(exist_ok=False);results=[]
try:
    for n in [4096,1<<20,16<<20]:
        src=P();assert alloc(C.byref(src),n)==0 and fill(src,0x5a,n)==0 and sync()==0
        for touched in [False,True]:
            path=temp/f'{n}-{touched}.bin';t=time.perf_counter_ns();fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.ftruncate(fd,n);region=mmap.mmap(fd,n);owner=(C.c_char*n).from_buffer(region);address=C.addressof(owner)
            if touched:C.memset(address,0,n)
            setup=time.perf_counter_ns()-t;t=time.perf_counter_ns();rc=register(address,n,0);reg=time.perf_counter_ns()-t;name=C.c_char_p();errname(rc,C.byref(name));row=dict(bytes=n,pretouched=touched,setup_ns=setup,register_ns=reg,register_rc=rc,register_name=name.value.decode())
            if rc==0:
                times=[]
                for _ in range(32):
                    t=time.perf_counter_ns();assert dtoh(address,src,n)==0;times.append(time.perf_counter_ns()-t)
                assert region[:]==b'Z'*n;t=time.perf_counter_ns();assert unregister(address)==0;row.update(unregister_ns=time.perf_counter_ns()-t,copy_ns=times,content_verified=True)
            results.append(row);del owner;region.close();os.close(fd);path.unlink()
        assert free(src)==0
finally:temp.rmdir()
out.write_text(json.dumps(dict(feasibility_only=True,results=results),indent=2)+'\n')
