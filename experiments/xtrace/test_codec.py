#!/usr/bin/env python3
"""Same-execution bytewise GPU round trips, range escapes and raw recovery."""
import ctypes as C
import hashlib
import json
import random
import struct
from pathlib import Path
def bind(lib,name,args,result):
    f=getattr(lib,name);f.argtypes=args;f.restype=result;return f

def main():
    lib=C.CDLL('/tmp/xtrace_gpu_collector_unit32.so');cuda=C.CDLL('libcudart.so.12')
    create=bind(lib,'xg_create',[C.c_int,C.c_int,C.POINTER(C.c_uint)],C.c_void_p)
    begin=bind(lib,'xg_begin',[C.c_void_p,C.c_char_p],C.c_int)
    submit=bind(lib,'xg_submit',[C.c_void_p,C.c_uint64,C.c_uint64],C.c_int)
    finish=bind(lib,'xg_finish',[C.c_void_p],C.c_int)
    validate=bind(lib,'xg_validate_file',[C.c_char_p,C.POINTER(C.c_uint),C.c_void_p,C.POINTER(C.c_uint64)],C.c_int)
    destroy=bind(lib,'xg_destroy',[C.c_void_p],None);error=bind(lib,'xg_error',[],C.c_char_p)
    malloc=bind(cuda,'cudaMalloc',[C.POINTER(C.c_void_p),C.c_size_t],C.c_int)
    copy=bind(cuda,'cudaMemcpy',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int],C.c_int)
    free=bind(cuda,'cudaFree',[C.c_void_p],C.c_int)
    widths=(C.c_uint*256)();widths[1]=8
    unit=bind(lib,'xg_ref_unit',[],C.c_uint64)()
    cases=[]
    rng=random.Random(20261012)
    for case in ['dense','unordered','local_escape','anchor_escape','grid64_raw','many_anchors_raw','chunk_boundary','unaligned']:
        n=4096 if case!='chunk_boundary' else 720000
        records=[]
        for i in range(n):
            g=1+(i//1024);w=i//2;sm=(w//8)%82;stamp=1000000000+g*10000+w*10+(i%2)*20
            if case=='local_escape':stamp+=(i%2)*100000
            if case=='anchor_escape':stamp+=g*100000000
            if case=='grid64_raw':g+=1<<40
            if case=='many_anchors_raw':g=i+1
            stamp*=unit
            if case=='unaligned':stamp+=i%7
            records.append(struct.pack('<QQIHBB',stamp,g,w,sm,1+i%2,1))
        if case=='unordered':rng.shuffle(records)
        raw=b''.join(records);ptr=C.c_void_p();assert malloc(C.byref(ptr),len(raw))==0
        buf=C.create_string_buffer(raw);assert copy(ptr,buf,len(raw),1)==0
        for mode in [0,1]:
            c=create(mode,1,widths);assert c,error().decode();path=Path(f'/tmp/xgpu-test-{case}-{mode}')
            try:
                assert begin(c,str(path).encode())==0,error().decode()
                assert submit(c,ptr.value,n)==0,error().decode()
                assert finish(c)==0,error().decode()
                h=(C.c_ubyte*32)();stats=(C.c_uint64*3)();rc=validate(str(path).encode(),widths,h,stats);assert rc==0,error().decode()
                assert bytes(h).hex()==hashlib.sha256(raw).hexdigest() and stats[0]==n
                cases.append({'case':case,'mode':mode,'records':n,'file_bytes':path.stat().st_size,'raw_sha256':bytes(h).hex(),'pass':True})
            finally:destroy(c);path.unlink(missing_ok=True)
        assert free(ptr)==0
    Path('/tmp/xtrace-paper-gpu-unit32/codec-tests.json').write_text(json.dumps(cases,indent=2));print(json.dumps({'tests':len(cases),'pass':True}),flush=True)

if __name__=='__main__':main()
