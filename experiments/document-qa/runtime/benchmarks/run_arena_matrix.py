"""Independent-process paired CUDA ablations. Raw capture stays in remote tmpfs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import struct
import subprocess
import sys
import time

MODES = ['passthrough','framework','buffers','pruned','probe','d2h','files','logged']
BASE = {f'P{i}': dict(mode=MODES[i-1], env={}) for i in range(1,9)}
BASE['P0'] = dict(mode=None, env={})


def trace_check(path, records=False):
    if path.suffix == '.nts':
        if not records:
            from arena_checks import sequence
            a=sequence(path)
            return dict(file=path.name,bytes=path.stat().st_size,frames=a['frames'],logical_bytes=a['payload']+48*a['frames'],record_validation=False,nonzero_records=0,zero_records=0,observed_sm_ids=[],format='NTGPU13')
        from trace_codec import check_gpu
        return check_gpu(path, records)
    if path.suffix == '.ntp':
        from trace_codec import check_packed
        return check_packed(path, records)
    if path.suffix == '.ntz':
        from trace_codec import check_compressed
        return check_compressed(path, records)
    return check_raw(path, records)


def check_raw(path, records=False):
    """Validate all concatenated frames in a selected file, without reading payload."""
    size = path.stat().st_size
    pos = frames = 0
    nonzero = zeros = logical_bytes = 0
    sm_ids = set()
    with path.open('rb') as fp:
        while pos < size:
            fp.seek(pos)
            raw = fp.read(32)
            if len(raw) != 32:
                raise RuntimeError(f'Short trace header: {path}')
            gx,gy,gz,bx,by,bz,shared,maps = struct.unpack('<8I',raw)
            if maps != 1 or not min(gx,gy,gz,bx,by,bz):
                raise RuntimeError(f'Invalid trace shape: {path}')
            item,div,offset = struct.unpack('<IIQ',fp.read(16))
            if div not in (1,32) or offset != 48 or item != 16:
                raise RuntimeError(f'Invalid block_sched section: {path}')
            payload = gx*gy*gz*((bx*by*bz+div-1)//div)*item
            logical_bytes += payload
            if records:
                data=fp.read(payload)
                if len(data)!=payload: raise RuntimeError('Short record payload')
                from trace_codec import record_stats
                nz,z,ids=record_stats(data);nonzero+=nz;zeros+=z;sm_ids.update(ids)
            pos += offset+payload
            if pos > size:
                raise RuntimeError(f'Trace exceeds file: {path}')
            frames += 1
    return dict(file=path.name, bytes=size, frames=frames,logical_bytes=logical_bytes,record_validation=records,nonzero_records=nonzero,
                zero_records=zeros,observed_sm_ids=sorted(sm_ids))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--trace-root',type=Path,default=Path('/dev/shm/neutrino-ablation'))
    ap.add_argument('--conditions',nargs='+',default=list(BASE))
    ap.add_argument('--spec',type=Path)
    ap.add_argument('--rounds',type=int,default=7)
    ap.add_argument('--seed',type=int,default=21042)
    ap.add_argument('--diagnostic',action='store_true')
    ap.add_argument('--compiler-python',type=Path,
                    help='optional cold compiler launcher; never used for workload Python')
    ap.add_argument('--timeout',type=float,default=600)
    ap.add_argument('command',nargs=argparse.REMAINDER)
    args=ap.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    specs=json.loads(args.spec.read_text()) if args.spec else BASE
    if not command or args.rounds<1 or any(c not in specs for c in args.conditions):
        ap.error('workload, positive rounds, and valid conditions required')
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    args.trace_root.mkdir(parents=True,exist_ok=True)
    order=list(args.conditions); random.Random(args.seed).shuffle(order)
    orders=[order[r%len(order):]+order[:r%len(order)] for r in range(args.rounds)]
    config=dict(command=command,specs=specs,conditions=args.conditions,orders=orders,
                seed=args.seed,rounds=args.rounds,diagnostic=args.diagnostic,
                timing=args.diagnostic,stats=args.diagnostic,trace_root=str(args.trace_root))
    config['compiler_python']=str(args.compiler_python.resolve()) if args.compiler_python else None
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    records=[]
    for r,sequence in enumerate(orders,1):
        for condition in sequence:
            spec=specs[condition]
            diagnostic=bool(spec.get('diagnostic',args.diagnostic))
            run=args.output/f'r{r}-{condition}'; run.mkdir()
            trace=args.trace_root/f'{args.output.name}-r{r}-{condition}'
            trace.mkdir(exist_ok=False)
            env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')}
            env['NEUTRINO_BENCH_DIR']=str(run)
            # Expose an empty trace directory to native controls too; the
            # workload's bounded-space validator then verifies zero capture.
            env['NEUTRINO_TRACEDIR']=str(trace)
            env['NEUTRINO_ABLATION_STATS']=str(int(diagnostic))
            if args.compiler_python:
                env['NEUTRINO_COMPILER_CACHE_ROOT']=os.environ['NEUTRINO_COMPILER_CACHE_ROOT']
            env.update({k:str(v) for k,v in spec.get('env',{}).items()})
            argv=command
            if spec['mode']:
                argv=[sys.executable,'-m','neutrino.cli','-p','block_sched',
                      '--callback','/dev/null','--tracedir',str(trace),
                      '--ablation-mode',spec['mode']]
                if diagnostic:
                    argv+=['--timing','--timing-dir',str(run)]
                if args.compiler_python:
                    argv+=['--python',str(args.compiler_python.resolve())]
                argv+=command
            (run/'command.json').write_text(json.dumps(argv,indent=2)+'\n')
            print(f'ROUND {r} {condition} START',flush=True)
            begin=time.perf_counter_ns()
            try:
                with (run/'stdout.log').open('wb') as stdout,(run/'stderr.log').open('wb') as stderr:
                    proc=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,start_new_session=True,cwd=spec.get('root'))
                    try:
                        rc=proc.wait(timeout=args.timeout)
                        if rc:
                            raise RuntimeError(f'{condition} exited {rc}; see {run}')
                    except BaseException:
                        try:
                            os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=5)
                        except ProcessLookupError:
                            pass
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                        raise
                w=json.loads((run/'workload.json').read_text())
                if len({(x['input_tokens'], x['output_tokens']) for x in w['requests']}) != 1:
                    raise RuntimeError('Unexpected token counts')
                if diagnostic and spec['mode']:
                    d=json.loads((run/'measurement-ablation.json').read_text())
                    if d['cold'] or any(g['fallback'] or g['errors'] for g in d['groups']):
                        raise RuntimeError('Cold/fallback/error in diagnostic measurement')
                    if not sum(g['calls'] for g in d['groups']):
                        raise RuntimeError('No captured launches')
                    t=json.loads((run/'measurement-timing.json').read_text())
                    if any(sum(s['total_ns'] for s in g['stages'].values())!=g['host_wall']['total_ns']
                           for g in t['groups']):
                        raise RuntimeError('Timing budget invalid')
                paths=list(trace.glob('*/result/*.bin'))+list(trace.glob('*/result/*.ntz'))+list(trace.glob('*/result/*.ntp'))+list(trace.glob('*/result/*.nts'))
                if spec.get('env',{}).get('NEUTRINO_NVPIPE_LIBRARY'):
                    # File validation runs in this parent, not in the workload
                    # subprocess that received the condition's environment.
                    os.environ['NEUTRINO_NVPIPE_LIBRARY']=spec['env']['NEUTRINO_NVPIPE_LIBRARY']
                    import trace_codec
                    trace_codec._nv_decoder=None
                if os.environ.get('ARENA_SAVE_LAYOUT')=='1' and paths:
                    with paths[0].open('rb') as src,(run/'frame-layout.tsv').open('w') as dest:
                        magic=src.read(8);assert magic in (b'NTGPU13\n',b'NTGPUV2\n') or magic[:4] in (b'NTFX',b'NTHR',b'NTRF',b'NTNV')
                        while True:
                            h=src.read(48)
                            if not h:break
                            dims=struct.unpack('<8I',h[:32]);packed,=struct.unpack('<Q',src.read(8));n=dims[0]*dims[1]*dims[2]*((dims[3]*dims[4]*dims[5]+31)//32)
                            size=(packed&((1<<62)-1)) or n*16
                            dest.write(' '.join(map(str,(*dims[:7],size,packed)))+'\n');src.seek(size,1)
                if os.environ.get('MECHANISM_CODEC_CONTROL')=='1' and paths:
                    last=max(paths,key=lambda p:int(p.stem.split('-')[-1]))
                    subprocess.run([sys.executable,str(Path(__file__).parent/'codec_recorded_control.py'),str(last),str(run/'codec-control')],check=True)
                counts=dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths))
                if os.environ.get('FINAL_QUERY_ANALYSIS')=='1' and paths:
                    from final_trace_queries import analyze_files
                    (run/'query-validation.json').write_text(json.dumps(analyze_files(paths),indent=2)+'\n')
                # Long-output runs can validate each closed request inside the
                # workload and delete its raw trace immediately to bound tmpfs.
                # Reuse those per-file checks here and prove the directory is
                # empty; never mistake a deleted trace for a native run.
                if w.get('trace_files_removed_after_validation'):
                    streamed=w.get('streamed_trace_validation',{})
                    assert streamed.get('enabled') and streamed.get('all_validated_before_delete')
                    assert not paths, f'streamed trace files unexpectedly remain: {paths[:3]}'
                    requests=streamed.get('requests',[])
                    assert len(requests)==w['warmup_requests']+len(w['requests'])
                    assert all(x.get('validated_before_delete') for x in requests)
                    if spec['mode']:
                        assert all(x['files']>0 and x['frames']>0 for x in requests), 'captured request has no validated frames'
                    else:
                        assert all(x['files']==0 and x['frames']==0 for x in requests), 'native workload unexpectedly captured trace'
                    measured=requests[-len(w['requests']):]
                    counts=dict(files=sum(x['files'] for x in measured),
                                bytes=sum(x['trace_bytes'] for x in measured))
                    checks=[c for x in measured for c in x['checks']]
                    total_frames=sum(x['frames'] for x in measured)
                    total_logical=sum(x['logical_bytes'] for x in measured)
                    if spec['mode']:
                        assert total_frames>0 and len(checks)==counts['files']
                        summary_path=run/'capture-summary.json'
                        if summary_path.exists():
                            summary=json.loads(summary_path.read_text())
                            for key in ('arena_engines','async_engines'):
                                engines=summary.get(key,[])
                                if engines:
                                    assert len(engines)==1
                                    engine=engines[0]
                                    if key=='async_engines':
                                        assert engine['submitted']==engine['completed']==total_frames
                                        assert engine['payload_bytes']==total_logical
                                        assert engine['trace_bytes']==counts['bytes']
                    from arena_checks import check_run
                    extra=dict(streamed_trace=True,frames=total_frames,logical_bytes=total_logical,
                               files=counts['files'],bytes=counts['bytes'],
                               requests_validated=len(requests),all_headers_checked=True,
                               all_files_deleted_after_validation=True)
                    checks=[dict(frames=total_frames,logical_bytes=total_logical,
                                 files=counts['files'],bytes=counts['bytes'],
                                 record_validation=False)]
                else:
                    # Only a small selected set is parsed; every retained sample is checked in full.
                    selected=list(dict.fromkeys(paths[:2]+paths[-2:]+sorted(paths,key=lambda p:p.stat().st_size)[-2:]))
                    checks=[trace_check(p,os.environ.get('BULK_RECORD_VALIDITY')=='1') for p in paths]
                    from arena_checks import check_run
                    if paths and all(p.suffix=='.bin' for p in paths):
                        total_frames=sum(c['frames'] for c in checks)
                        requests=w['warmup_requests']+len(w['requests'])
                        assert total_frames>0 and total_frames%requests==0
                        extra=dict(legacy_sync=True,frames=total_frames,frames_per_request=total_frames//requests,files=len(paths),bytes=counts['bytes'],all_headers_checked=True)
                        checks=[dict(frames=total_frames,files=len(paths),bytes=counts['bytes'],record_validation=False)]
                    else:
                        extra=check_run(run, trace, paths, checks, w, bool(spec['mode']))
                (run/'arena-check.json').write_text(json.dumps(extra,indent=2)+'\n')
                compiler=[]
                caches=[json.loads(p.read_text()) for p in trace.glob('*/kernel/*/compile-cache.json')]
                (run/'compile-cache.json').write_text(json.dumps(caches,indent=2)+'\n')
                if diagnostic:
                    for path in trace.glob('*/kernel/*/process-timing.json'):
                        compiler.append(dict(directory=path.parent.name,**json.loads(path.read_text())))
                    resources=[]
                    for path in trace.glob('*/kernel/*/process.log'):
                        resources.append(dict(directory=path.parent.name,ptxas_log=path.read_text()))
                    (run/'compiler-resources.json').write_text(json.dumps(resources,indent=2)+'\n')
                    # Static resources and SASS from both actual assembled versions.
                    for path in trace.glob('*/kernel/*/pruned.bin'):
                        name=(path.parent/'kernel.info').read_text().splitlines()[0]
                        if False:
                            dest=run/'sass'/path.parent.name; dest.mkdir(parents=True,exist_ok=True)
                            for kind in ('pruned','probed'):
                                binpath=path.parent/f'{kind}.bin'
                                result=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage','--dump-sass',str(binpath)],capture_output=True,text=True,check=True)
                                (dest/f'{kind}.txt').write_text(result.stdout)
                (run/'compiler-timing.json').write_text(json.dumps(compiler,indent=2)+'\n')
                rec=dict(round=r,condition=condition,process_wall_ns=time.perf_counter_ns()-begin,
                         trace=counts,trace_checks=checks,**w)
                records.append(rec)
                (run/'run.json').write_text(json.dumps(rec,indent=2)+'\n')
                if len({x['output_digest'] for x in records})!=1:
                    raise RuntimeError('Output digest differs')
                print(f'ROUND {r} {condition} DONE {w["wall_ns"]/len(w["requests"])/1e6:.3f} ms/request',flush=True)
            finally:
                shutil.rmtree(trace)
            (args.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
    (args.output/'complete.json').write_text(json.dumps(dict(processes=len(records),digest=records[0]['output_digest']))+'\n')


if __name__=='__main__':
    main()
