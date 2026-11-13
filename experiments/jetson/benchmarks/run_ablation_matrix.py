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
    """Validate all concatenated frames in a selected file, without reading payload."""
    size = path.stat().st_size
    pos = frames = 0
    nonzero = zeros = 0
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
            if records and payload >= 1000000:
                data=fp.read(payload)
                if len(data)!=payload: raise RuntimeError('Short record payload')
                for start,elapsed,smid in struct.iter_unpack('<QII',data):
                    if start or elapsed or smid: nonzero+=1;sm_ids.add(smid)
                    else: zeros+=1
            pos += offset+payload
            if pos > size:
                raise RuntimeError(f'Trace exceeds file: {path}')
            frames += 1
    return dict(file=path.name, bytes=size, frames=frames,record_validation=records,nonzero_records=nonzero,
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
            run=args.output/f'r{r}-{condition}'; run.mkdir()
            trace=args.trace_root/f'{args.output.name}-r{r}-{condition}'
            trace.mkdir(exist_ok=False)
            env={k:v for k,v in os.environ.items() if not k.startswith('NEUTRINO_')}
            env['NEUTRINO_BENCH_DIR']=str(run)
            env['NEUTRINO_ABLATION_STATS']=str(int(args.diagnostic))
            if args.compiler_python:
                env['NEUTRINO_COMPILER_CACHE_ROOT']=os.environ['NEUTRINO_COMPILER_CACHE_ROOT']
            env.update({k:str(v) for k,v in spec.get('env',{}).items()})
            argv=command
            if spec['mode']:
                argv=[sys.executable,'-m','neutrino.cli','-p','block_sched',
                      '--callback','/dev/null','--tracedir',str(trace),
                      '--ablation-mode',spec['mode']]
                if args.diagnostic:
                    argv+=['--timing','--timing-dir',str(run)]
                if args.compiler_python:
                    argv+=['--python',str(args.compiler_python.resolve())]
                argv+=command
            (run/'command.json').write_text(json.dumps(argv,indent=2)+'\n')
            print(f'ROUND {r} {condition} START',flush=True)
            begin=time.perf_counter_ns()
            try:
                with (run/'stdout.log').open('wb') as stdout,(run/'stderr.log').open('wb') as stderr:
                    proc=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
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
                if any(x['input_tokens']!=39 or x['output_tokens']!=32 for x in w['requests']):
                    raise RuntimeError('Unexpected token counts')
                if args.diagnostic and spec['mode']:
                    d=json.loads((run/'measurement-ablation.json').read_text())
                    if d['cold'] or any(g['fallback'] or g['errors'] for g in d['groups']):
                        raise RuntimeError('Cold/fallback/error in diagnostic measurement')
                    if sum(g['calls'] for g in d['groups'])!=16648*len(w['requests']):
                        raise RuntimeError('Launch count differs')
                    t=json.loads((run/'measurement-timing.json').read_text())
                    if any(sum(s['total_ns'] for s in g['stages'].values())!=g['host_wall']['total_ns']
                           for g in t['groups']):
                        raise RuntimeError('Timing budget invalid')
                paths=list(trace.glob('*/result/*.bin'))
                counts=dict(files=len(paths),bytes=sum(p.stat().st_size for p in paths))
                # Only a small selected set is parsed; every retained sample is checked in full.
                selected=list(dict.fromkeys(paths[:2]+paths[-2:]+sorted(paths,key=lambda p:p.stat().st_size)[-2:]))
                checks=[trace_check(p,os.environ.get('NEUTRINO_RECORD_VALIDITY')=='1') for p in selected] if args.diagnostic else []
                compiler=[]
                caches=[json.loads(p.read_text()) for p in trace.glob('*/kernel/*/compile-cache.json')]
                (run/'compile-cache.json').write_text(json.dumps(caches,indent=2)+'\n')
                if args.diagnostic:
                    for path in trace.glob('*/kernel/*/process-timing.json'):
                        compiler.append(dict(directory=path.parent.name,**json.loads(path.read_text())))
                    resources=[]
                    for path in trace.glob('*/kernel/*/process.log'):
                        resources.append(dict(directory=path.parent.name,ptxas_log=path.read_text()))
                    (run/'compiler-resources.json').write_text(json.dumps(resources,indent=2)+'\n')
                    # Static resources and SASS from both actual assembled versions.
                    for path in trace.glob('*/kernel/*/pruned.bin'):
                        name=(path.parent/'kernel.info').read_text().splitlines()[0]
                        if any(x in name for x in ('quantize_q8_1','rms_norm_f32','ggml_type14ELi1ELb0ELb0ELb0')):
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
