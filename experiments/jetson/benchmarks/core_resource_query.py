"""Query physical resources/occupancy for actual Qwen shapes; no model launches."""
import ctypes as C,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];cu=C.CDLL('libcuda.so.1');H=C.c_void_p
def bind(n,args):
    f=getattr(cu,n);f.argtypes=args;f.restype=C.c_int;return f
init=bind('cuInit',[C.c_uint]);dg=bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);retain=bind('cuDevicePrimaryCtxRetain',[C.POINTER(H),C.c_int]);setctx=bind('cuCtxSetCurrent',[H]);load=bind('cuModuleLoad',[C.POINTER(H),C.c_char_p]);get=bind('cuModuleGetFunction',[C.POINTER(H),H,C.c_char_p]);unload=bind('cuModuleUnload',[H]);attr=bind('cuFuncGetAttribute',[C.POINTER(C.c_int),C.c_int,H]);setattr_=bind('cuFuncSetAttribute',[H,C.c_int,C.c_int]);occ=bind('cuOccupancyMaxActiveBlocksPerMultiprocessor',[C.POINTER(C.c_int),H,C.c_int,C.c_size_t])
assert init(0)==0;dev=C.c_int();ctx=H();assert dg(C.byref(dev),0)==0 and retain(C.byref(ctx),dev)==0 and setctx(ctx)==0
shapes={}
for p in (root/'results/core-gpu').glob('r1-*/cupti.json'):
    for k in json.loads(p.read_text())['kernels']:
        if not k['name'].startswith('neutrino_'):shapes.setdefault(k['name'],set()).add(tuple(k['shape'][3:]))
rows=[]
for manifest in (root/'compiler-cache').glob('*/manifest.json'):
    d=json.loads(manifest.read_text());name=d['key_data']['kernel'];mode=d['key_data']['env']['NEUTRINO_CORE_MODE']
    matches=[key for key in shapes if key==name or (len(key)==511 and name.startswith(key))]
    assert len(matches)<=1
    if not matches:continue
    for kind in ['pruned','probed']:
        mod=H();f=H();assert load(C.byref(mod),str(manifest.parent/(kind+'.bin')).encode())==0 and get(C.byref(f),mod,name.encode())==0
        for bx,by,bz,shared in shapes[matches[0]]:
            rc=setattr_(f,8,shared) if shared else 0
            active=C.c_int();oreg=occ(C.byref(active),f,bx*by*bz,shared)
            attrs={}
            for n,index in [('registers',4),('local_bytes',3),('static_shared_bytes',1)]:
                v=C.c_int();assert attr(C.byref(v),index,f)==0;attrs[n]=v.value
            rows.append(dict(mode=mode,kernel=name,kind=kind,block=[bx,by,bz],dynamic_shared=shared,attribute_rc=rc,occupancy_rc=oreg,max_blocks_per_sm=active.value if not oreg else None,**attrs))
        assert unload(mod)==0
(root/'results/qwen-resources.json').write_text(json.dumps(rows,indent=2)+'\n');print('RESOURCE CONFIGURATIONS',len(rows))
