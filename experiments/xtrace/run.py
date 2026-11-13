#!/usr/bin/env python3
"""Paper task and completion boundary; matched Sparse / GPU ERE collectors."""
import argparse
import ctypes as C
import hashlib
import json
import os
import time
from pathlib import Path
from gwatch.cuda.profile import ProfileContext
from gwatch.cuda.binary import Cubin
from gwatch.cuda.xtrace._runner import find_kernel_ptx
from producer import bind, launch_warps, instrument_ptx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sparse', 'ere'], required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--requests', type=int, default=3)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--retain-last', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    ctx = ProfileContext()
    lib = C.CDLL(os.environ['QWEN_ADAPTER'])
    create = bind(lib,'qwen_create',[C.c_char_p,C.c_char_p],C.c_void_p)
    destroy = bind(lib,'qwen_destroy',[C.c_void_p],None)
    fill = bind(lib,'qwen_fill_token_ids',[C.c_void_p,C.c_int,C.POINTER(C.c_int)],C.c_int)
    prefill = bind(lib,'qwen_prefill_ids',[C.c_void_p,C.POINTER(C.c_int),C.c_int],C.c_int)
    argmax = bind(lib,'qwen_argmax',[C.c_void_p],C.c_int)
    step = bind(lib,'qwen_step',[C.c_void_p,C.c_int,C.c_int],C.c_int)
    cuda = C.CDLL('libcudart.so.12')
    sync = bind(cuda,'cudaDeviceSynchronize',[],C.c_int)
    malloc = bind(cuda,'cudaMalloc',[C.POINTER(C.c_void_p),C.c_size_t],C.c_int)
    free = bind(cuda,'cudaFree',[C.c_void_p],C.c_int)
    memset = bind(cuda,'cudaMemset',[C.c_void_p,C.c_int,C.c_size_t],C.c_int)
    codec = C.CDLL(os.environ['XTRACE_GPU_LIB'])
    error = bind(codec,'xg_error',[],C.c_char_p)
    new = bind(codec,'xg_create',[C.c_int,C.c_int,C.POINTER(C.c_uint)],C.c_void_p)
    beginfile = bind(codec,'xg_begin',[C.c_void_p,C.c_char_p],C.c_int)
    submit = bind(codec,'xg_submit',[C.c_void_p,C.c_uint64,C.c_uint64],C.c_int)
    count = bind(codec,'xg_count',[C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)],C.c_int)
    finish = bind(codec,'xg_finish',[C.c_void_p],C.c_int)
    stats = bind(codec,'xg_stats',[C.c_void_p,C.POINTER(C.c_uint64)],None)
    hashes = bind(codec,'xg_hashes',[C.c_void_p,C.c_void_p],None)
    delete = bind(codec,'xg_destroy',[C.c_void_p],None)
    snapshot_bytes=bind(codec,'xg_snapshot_bytes',[],C.c_uint64)
    source_hash=bind(codec,'xg_source_hash',[C.c_uint64,C.c_uint64,C.c_void_p],C.c_int)
    compare_source=bind(codec,'xg_compare_source',[C.c_uint64,C.c_char_p,C.POINTER(C.c_uint)],C.c_int)
    validate = bind(codec,'xg_validate_file',[C.c_char_p,C.POINTER(C.c_uint),C.c_void_p,C.POINTER(C.c_uint64)],C.c_int)
    def ck(rc):
        if rc: raise RuntimeError(error().decode())
    handle = create(os.environ['QWEN_MODEL'].encode(),os.environ['QWEN_BACKEND'].encode())
    if not handle: raise RuntimeError('model creation failed')
    ptr = C.c_void_p(); collector = None; registered = [];replacements=[]
    try:
        tokens = (C.c_int*256)(); assert fill(handle,256,tokens)==256
        def infer(flush=None):
            assert prefill(handle,tokens,256)==256
            generated=[argmax(handle)]
            if flush: flush()
            for index in range(63):
                assert step(handle,generated[-1],256+index)==0
                generated.append(argmax(handle))
                if flush: flush()
            assert sync()==0
            return hashlib.sha256(json.dumps([generated],separators=(',',':')).encode()).hexdigest()
        ctx.start_tracing_kernel_launch(); scout_digest=infer(); scout=ctx.stop_tracing_kernel_launch()
        names=sorted({x.definition.mangled_prototype for x in scout})
        def signature(items):
            return [(x.definition.mangled_prototype,x.grid_dim_x,x.grid_dim_y,x.grid_dim_z,
                     x.block_dim_x,x.block_dim_y,x.block_dim_z) for x in items
                    if x.definition.mangled_prototype in names]
        expected_signature=signature(scout)
        expected=2*sum(launch_warps(x) for x in scout)
        capacity=int(expected*1.02)+1024; allocation=16+capacity*24
        assert malloc(C.byref(ptr),allocation)==0
        widths=(C.c_uint*256)()
        for kid,name in enumerate(names,1):
            found={((x.block_dim_x*x.block_dim_y*x.block_dim_z+31)//32) for x in scout if x.definition.mangled_prototype==name}
            if len(found)!=1: raise RuntimeError('CTA shape not unique for kernel')
            widths[kid]=found.pop()
        collector=new(a.mode=='ere',a.verify,widths)
        if not collector: raise RuntimeError(error().decode())
        bins=ctx.get_list_binary_ptx(); compiled=[]
        cache=Path(os.environ.get('XTRACE_CACHE','/tmp/xtrace-paper-gpu-cache'));cache.mkdir(exist_ok=True)
        producer_hash=hashlib.sha256(Path(__file__).with_name('producer.py').read_bytes()).hexdigest()
        for kid,name in enumerate(names,1):
            key=hashlib.sha256(f'{producer_hash}:{capacity}:{kid}:{name}'.encode()).hexdigest()
            path=cache/(key+'.cubin')
            if path.exists(): image=Cubin();image.fill(str(path));sites=2
            else:
                binary=find_kernel_ptx(name,'',bins)
                if binary is None: raise RuntimeError('PTX missing: '+name)
                image,sites=instrument_ptx(binary,name,kid,capacity);image.dump(str(path))
            ctx.load_binary_cubin(binary_cubin=image)
            replacement=ctx.load_kerneldef_sass(binary_cubin=image,list_load_kernel_name=[name])[0]
            original=ctx.get_kernel_def_sass_by_name(name)
            if original is None: raise RuntimeError('SASS def missing: '+name)
            ctx.register_trace_kernel_sass(original,replacement,param_values=[ptr.value]);registered.append(original);replacements.append((original,replacement))
            compiled.append({'id':kid,'name':name,'sites':sites,'warps_per_cta':widths[kid],'cubin_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            print(json.dumps({'phase':'compiled','index':kid,'total':len(names)}),flush=True)
        measured=[]; warm=[]
        for index in range(a.warmup+a.requests):
            if not registered:
                for original,replacement in replacements:
                    ctx.register_trace_kernel_sass(original,replacement,param_values=[ptr.value]);registered.append(original)
            assert memset(ptr,0,16)==0
            is_warm=index<a.warmup; path=a.out/f'{"warm" if is_warm else "request"}-{index+1}.xgpu'
            counter=C.c_uint64()
            def flush():
                # Reservation count is published before record stores.
                # Fence ALL replacement/producer streams before consuming a prefix.
                assert sync()==0
                ck(count(collector,ptr.value,C.byref(counter)))
                if counter.value>capacity: raise RuntimeError('capture overflow')
                ck(submit(collector,ptr.value+16,counter.value))
            ctx.start_tracing_kernel_launch()
            t0=time.perf_counter_ns();ck(beginfile(collector,str(path).encode()))
            digest=infer();launches=ctx.stop_tracing_kernel_launch()
            for original in reversed(registered):ctx.unregister_trace_kernel_sass(original)
            registered.clear();infer_done=time.perf_counter_ns()
            committed=C.c_uint64();ready_deadline=time.monotonic()+60
            while True:
                ck(count(collector,ptr.value+8,C.byref(committed)))
                if committed.value==expected:break
                if committed.value>expected or time.monotonic()>ready_deadline:raise RuntimeError('committed record count did not reach expected complete scope')
            flush();ck(finish(collector));end=time.perf_counter_ns()
            actual_signature=signature(launches)
            if actual_signature!=expected_signature or len(launches)!=len(scout) or digest!=scout_digest or counter.value!=expected:
                failure={'stage':'warm' if is_warm else 'measured','index':index,'count':counter.value,'expected_count':expected,
                    'digest':digest,'scout_digest':scout_digest,'signature_equal':actual_signature==expected_signature,
                    'actual_launches':len(actual_signature),'expected_launches':len(expected_signature),
                    'first_differences':[(i,x,y) for i,(x,y) in enumerate(zip(actual_signature,expected_signature)) if x!=y][:5],
                    'trace_path':str(path),'request_ms':(end-t0)/1e6}
                (a.out/'failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps({'phase':'failed',**failure}),flush=True)
                raise RuntimeError('launch sequence, token output, or event count differs')
            v=(C.c_uint64*10)();stats(collector,v)
            fields=['records','trace_bytes','frames','encoded_frames','raw_frames','recoveries','cta_anchors','codec_scratch_bytes','device_output_pool_bytes','pinned_host_pool_bytes']
            row=dict(zip(fields,map(int,v)));row['device_snapshot_pool_bytes']=int(snapshot_bytes());row.update(request_ms=(end-t0)/1e6,trace_drain_ms=(end-infer_done)/1e6,
                output_digest=digest,launches=len(actual_signature),trace_path=str(path),file_close_in_timing=True)
            if path.stat().st_size!=row['trace_bytes']:raise RuntimeError('file size mismatch')
            if a.verify:
                h=(C.c_ubyte*64)();hashes(collector,h)
                hs=bytes(h);fh=(C.c_ubyte*32)();fs=(C.c_uint64*3)()
                original=(C.c_ubyte*32)();ck(source_hash(ptr.value+16,expected,original))
                if bytes(original)!=hs[:32]:
                    repeats=[bytes(original).hex()]
                    for _ in range(2):
                        ck(source_hash(ptr.value+16,expected,original));repeats.append(bytes(original).hex())
                    rc=compare_source(ptr.value+16,str(path).encode(),widths)
                    detail=error().decode() if rc else None
                    failure={'detail':detail,'phase':'source_read_integrity_failure','snapshot_sha256':hs[:32].hex(),'completed_source_repeat_sha256':repeats,'all_same':len(set(repeats))==1}
                    (a.out/'source_integrity_failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure),flush=True)
                    raise RuntimeError('immutable frames differ from full completed original source')
                ck(validate(str(path).encode(),widths,fh,fs))
                if hs[:32]!=hs[32:] or bytes(fh)!=hs[:32] or fs[0]!=expected:raise RuntimeError('file recovery hash mismatch')
                row.update(raw_sha256=hs[:32].hex(),decoded_sha256=bytes(fh).hex(),bytewise_same_execution_check=True,full_completed_source_sha256=bytes(original).hex())
            (warm if is_warm else measured).append(row)
            print(json.dumps({'phase':'warm' if is_warm else 'measured',**row}),flush=True)
            (a.out/'progress.json').write_text(json.dumps({'warmup':warm,'requests':measured},indent=2))
            if not(a.retain_last and index==a.warmup+a.requests-1):path.unlink()
        result={'mode':a.mode,'model':os.environ['QWEN_MODEL'],'task':{'input':256,'output':64,'batch':1,'context':4096,'graph':False,'threads':8,'eos_early_stop':False,'scope':'all hook-covered launches; warp entry and exit'},
            'time_definition':'CPU wall clock before prefill through application sync, GPU codec/D2H, complete tmpfs file close; excludes setup/JIT/scout and offline verification',
            'warmup':warm,'requests':measured,'mean_request_ms':sum(r['request_ms'] for r in measured)/len(measured),
            'mean_trace_bytes':sum(r['trace_bytes'] for r in measured)/len(measured),'producer_device_bytes':allocation,
            'compiled':compiled,'verify_mode':a.verify,'widths':list(widths),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cu')},
            'reference_unit_ns':int(bind(codec,'xg_ref_unit',[],C.c_uint64)()),'codec_binary_sha256':hashlib.sha256(Path(os.environ['XTRACE_GPU_LIB']).read_bytes()).hexdigest(),
            'engineering_note':'Global append producer retained from full XTrace adapter; ERE is GPU-side and D2H/file collection is pipelined after the complete inference request. This is an XTrace adapter, not the original Neutrino producer.'}
        (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'phase':'done','mean_ms':result['mean_request_ms'],'trace_bytes':result['mean_trace_bytes']}),flush=True)
    finally:
        for original in reversed(registered):ctx.unregister_trace_kernel_sass(original)
        if collector:delete(collector)
        if ptr.value:free(ptr)
        destroy(handle)

if __name__=='__main__':main()
