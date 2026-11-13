"""Serial low-level diagnostics; raw syscall/capture data is temporary."""
import collections
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from run_ablation_matrix import BASE, trace_check

root=Path(__file__).resolve().parents[1]
out=root/'results/deep';out.mkdir(parents=True,exist_ok=True)
specs={**BASE,**json.loads((root/'benchmarks/ablation_specs.json').read_text())}
py=sys.executable
workload=[py,'benchmarks/deep_workload.py','--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
 '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
 '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
 '--requests','1','--warmup','2','--output-tokens','32']
strace='/root/autodl-tmp/neutrino-deep-tools-20261003/strace/usr/bin/strace'

def syscall_summary(path):
    # -s0 excludes payload values. fd1000 marker writes delimit the window.
    rows={};pending={};started=False;ended=False;partial=0;begin=end=None
    ioctls=collections.Counter();writes=collections.Counter();threads=collections.Counter();by_thread={};sinks=collections.Counter();main_tid=None
    for line in path.open(errors='replace'):
        m=re.match(r'\s*(\d+)\s+(\d+\.\d+)\s+(.*)',line)
        if not m:continue
        tid,stamp,body=m.groups();stamp=float(stamp)
        if body.startswith(('write(1000<','write(1000,')):
            if ', 12)' in body:started=True;begin=stamp;main_tid=tid;continue
            if ', 10)' in body:ended=True;end=stamp;break
        if not started:continue
        if '<unfinished ...>' in body:
            pending[tid]=body.replace('<unfinished ...>','');continue
        if body.startswith('<... '):
            old=pending.pop(tid,None)
            if old is None:partial+=1;continue
            body=old+re.sub(r'^<\.\.\. \w+ resumed>', '',body)
        dm=re.search(r'<(\d+\.\d+)>$',body)
        duration=float(dm.group(1)) if dm else 0
        clean=re.sub(r'\s*<\d+\.\d+>$','',body)
        m=re.match(r'(\w+)\((.*)\)\s+=\s+(.*)$',clean)
        if not m:continue
        name,args,result=m.groups()
        row=rows.setdefault(name,dict(calls=0,duration_s=0,errors=0))
        row['calls']+=1;row['duration_s']+=duration;row['errors']+=result.lstrip().startswith('-1')
        threads[tid]+=1
        tr=by_thread.setdefault(tid,{}).setdefault(name,dict(calls=0,duration_s=0,errors=0))
        tr['calls']+=1;tr['duration_s']+=duration;tr['errors']+=result.lstrip().startswith('-1')
        if name=='ioctl':
            fields=args.split(',');ioctls[fields[1].strip() if len(fields)>1 else '?']+=1
        if name in ('write','writev'):
            n=re.match(r'\s*(\d+)',result)
            if n:writes[n.group(1)]+=1
            target=args.split(',',1)[0]
            sink='tmpfs_trace' if '/dev/shm/neutrino-deep/' in target else ('eventfd' if 'eventfd' in target else target)
            sinks[sink]+=1
    assert started and ended,'Missing measured-window markers'
    return dict(begin=begin,end=end,window_s=end-begin,syscalls=rows,ioctl_requests=dict(ioctls),
                write_result_bytes_histogram=dict(writes),thread_syscall_counts=dict(threads),
                main_tid=main_tid,syscalls_by_thread=by_thread,write_targets=dict(sinks),
                unmatched_resumed=partial,unfinished_at_end=len(pending),raw_retained=False)

def run(kind,condition):
    run=out/kind/condition
    if (run/'complete.json').exists():return
    run.mkdir(parents=True,exist_ok=True)
    trace=Path('/dev/shm/neutrino-deep')/f'{kind}-{condition}';trace.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NEUTRINO_','DEEP_'))}
    env.update(NEUTRINO_BENCH_DIR=str(run),NEUTRINO_ABLATION_STATS='1',
               NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache',
               PATH='/usr/local/cuda/bin:'+env.get('PATH',''))
    spec=specs[condition];env.update(spec['env'])
    command=list(workload)
    if kind.startswith('cupti'):
        env.update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY=str(int('activity' in kind)),
                   NEUTRINO_PROFILE_COMPAT='1')
    if kind.startswith('syscalls'):env['DEEP_MARKERS']='1'
    if spec['mode']:
        command=[py,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null',
                 '--ablation-mode',spec['mode'],'--tracedir',str(trace),
                 '--python',str(root/'benchmarks/cache_python.py'),*command]
    raw=Path('/dev/shm')/f'neutrino-syscalls-{os.getpid()}-{condition}.txt'
    if kind.startswith('syscalls'):
        command=[strace,'-f','-ttt','-T','-yy','-s','0','-e',
                 'trace=read,write,writev,ioctl,futex,sched_yield,poll,ppoll,openat,close,mmap,munmap,brk,unlink,clock_nanosleep',
                 '-o',str(raw),*command]
    (run/'config.json').write_text(json.dumps(dict(command=command,env={k:v for k,v in env.items() if k.startswith(('NEUTRINO_','DEEP_'))}),indent=2)+'\n')
    print('DEEP START',kind,condition,flush=True)
    try:
        with (run/'process.log').open('w') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=log,check=True,timeout=600)
        w=json.loads((run/'workload.json').read_text())
        assert w['output_digest']=='0be37a842462e3d71e853059945490c9fb052af64a530092d6b44dfac641298a'
        if spec['mode']:
            d=json.loads((run/'measurement-ablation.json').read_text())
            assert not d['cold'] and not any(g['fallback'] or g['errors'] for g in d['groups'])
            assert sum(g['calls'] for g in d['groups'])==16648
        if kind.startswith('cupti'):
            c=json.loads((run/'cupti.json').read_text());assert not c['overflow'] and not c['dropped']
            assert c['driver_callbacks']>0
            if 'activity' in kind:
                assert sum(g['count'] for g in c['kernels'])==16704
                assert c['copy_activity_count']>=224 and not c['unmatched_copies']
                if c.get('correlation_schema') == 2:
                    assert not c['ambiguous_copies']
                    assert sum(g['gpu_pieces'] for g in c['copy_groups']) == c['copy_activity_count']
                    assert sum(g['bytes'] for g in c['copy_groups']) == c['copy_activity_bytes']
                    assert all(g['api'].startswith('cuMemcpy') for g in c['copy_groups'])
                    assert all('Async' in g['api'] or g['callback_suffix_ns'] >= -1000*g['calls'] for g in c['copy_groups'])
                assert c['parse_status']==12 # CUPTI_ERROR_MAX_LIMIT_REACHED: buffer exhausted normally.
        else:
            summary=syscall_summary(raw);(run/'syscalls.json').write_text(json.dumps(summary,indent=2)+'\n')
        paths=list(trace.glob('*/result/*.bin'))
        selected=list(dict.fromkeys(paths[:1]+sorted(paths,key=lambda p:p.stat().st_size)[-1:]))
        result=dict(wall_ns=w['wall_ns'],trace_files=len(paths),trace_bytes=sum(p.stat().st_size for p in paths),
                    trace_checks=[trace_check(p) for p in selected],passed=True)
        (run/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
        print('DEEP DONE',kind,condition,w['wall_ns']/1e6,'ms',flush=True)
    finally:
        shutil.rmtree(trace)
        if raw.exists():raw.unlink()

if __name__=='__main__':
    for c in ['P0','S0','S1','S6','S4','S3','F0','F1','F2','F4','D0','D1','D3','B1','O1']:
        run('cupti-callback',c)
    for c in ['P0','P3','P4','P5','F0','S1','S6','S4','D3','O1']:
        run('cupti-activity',c)
    for c in ['S0','S1','S6','S3','F0','F1','F2','F4','O1']:
        run('syscalls',c)
    (out/'diagnostics-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
