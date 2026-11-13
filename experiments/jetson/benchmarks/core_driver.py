"""Small checked Driver API wrapper for independent research fixtures."""
import ctypes as C

P=C.c_uint64;H=C.c_void_p
class Driver:
    def __init__(self):
        self.lib=C.CDLL('libcuda.so.1')
        def bind(name,args):
            f=getattr(self.lib,name);f.argtypes=args;f.restype=C.c_int;return f
        self.init=bind('cuInit',[C.c_uint]);self.device=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int])
        self.retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(H),C.c_int]);self.setctx=bind('cuCtxSetCurrent',[H])
        self.alloc=bind('cuMemAlloc_v2',[C.POINTER(P),C.c_size_t]);self.free=bind('cuMemFree_v2',[P])
        self.htod=bind('cuMemcpyHtoD_v2',[P,H,C.c_size_t]);self.dtoh=bind('cuMemcpyDtoH_v2',[H,P,C.c_size_t])
        self.hostalloc=bind('cuMemAllocHost_v2',[C.POINTER(H),C.c_size_t]);self.hostfree=bind('cuMemFreeHost',[H])
        self.load=bind('cuModuleLoad',[C.POINTER(H),C.c_char_p]);self.get=bind('cuModuleGetFunction',[C.POINTER(H),H,C.c_char_p])
        self.launch=bind('cuLaunchKernel',[H,*([C.c_uint]*7),H,C.POINTER(H),C.POINTER(H)])
        self.sync=bind('cuCtxSynchronize',[]);self.attr=bind('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,H])
        self.occ=bind('cuOccupancyMaxActiveBlocksPerMultiprocessor',[C.POINTER(C.c_int),H,C.c_int,C.c_size_t])
        self.ok(self.init(0));dev=C.c_int();ctx=H();self.ok(self.device(C.byref(dev),0));self.ok(self.retain(C.byref(ctx),dev));self.ok(self.setctx(ctx))
        self.modules=[]
    @staticmethod
    def ok(code):
        if code:raise RuntimeError(f'CUDA error {code}')
    def memory(self,n):
        p=P();self.ok(self.alloc(C.byref(p),max(n,1)));return p
    def pinned(self,n):
        p=H();self.ok(self.hostalloc(C.byref(p),max(n,1)));return p
    def upload(self,p,data):self.ok(self.htod(p,data,len(data)))
    def read(self,p,n):
        b=C.create_string_buffer(n);self.ok(self.dtoh(b,p,n));return b.raw
    def function(self,path,name):
        m=H();f=H();self.ok(self.load(C.byref(m),str(path).encode()));self.ok(self.get(C.byref(f),m,name.encode()));self.modules.append(m);return f
    def call(self,f,grid,block,args):
        ar=(H*len(args))(*(C.addressof(x) for x in args));self.ok(self.launch(f,*grid,*block,0,None,ar,None))
    def resources(self,f,threads):
        r={}
        for n,i in [('registers',4),('local_bytes',3),('shared_bytes',1)]:
            v=C.c_int();self.ok(self.attr(C.byref(v),i,f));r[n]=v.value
        v=C.c_int();self.ok(self.occ(C.byref(v),f,threads,0));r['blocks_per_sm']=v.value;return r
