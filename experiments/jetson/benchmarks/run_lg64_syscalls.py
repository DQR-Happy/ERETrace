"""Marked syscall and user-space write-stack diagnostics, after performance runs."""
import collections
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from run_deep_suite import syscall_summary
from run_bulk_matrix import trace_check

root=Path(__file__).resolve().parents[1];out=root/'results/syscalls';out.mkdir(parents=True,exist_ok=True)
specs=json.loads((root/'results/specs.json').read_text());py=sys.executable
tool='/root/autodl-tmp/neutrino-deep-tools-20261003/strace/usr/bin/strace'
work=[py,str(root/'benchmarks/qwen_capture_workload.py'),'--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
    '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin','--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
    '--requests','1','--warmup','2','--output-tokens','32']
def stacks(path):
    started=False;end=False;counter=collections.Counter();current=[];examples=[]
    for line in path.open(errors='replace'):
        if re.search(r'write\(1000[<,]',line):
            if ', 12)' in line:started=True
            if ', 10)' in line:end=True;break
            continue
        if not started:continue
        if re.match(r'\s*\d+\s+\d+\.\d+\s+',line):
            if current and len(examples)<5:examples.append(current)
            current=[]
        elif ' > ' in line or line.lstrip().startswith('>'):
            text=line.split('>',1)[1].strip();text=re.sub(r'\s*\[0x[0-9a-f]+\]','',text)
            counter[text]+=1;current.append(text)
    if current and len(examples)<5:examples.append(current)
    assert started and end
    return dict(symbol_counts=dict(counter),representative_chains=examples,
                scope='user-space unwind at traced write syscalls; counts are not CPU samples',raw_retained=False)
for name,c,stack in [('N','N',False),('G','G',False),('AZ','AZ',False),('X','X',False),('G-write-stacks','G',True)]:
    dest=out/name
    if (dest/'complete.json').exists():continue
    dest.mkdir();trace=Path('/dev/shm/neutrino-lg64-detail')/('syscall-'+name);trace.mkdir(exist_ok=False)
    raw=Path('/dev/shm')/f'lg64-syscall-{os.getpid()}-{name}.txt'
    env={k:v for k,v in os.environ.items() if not k.startswith(('NEUTRINO_','DEEP_'))}
    env.update(PATH='/usr/local/cuda/bin:'+env.get('PATH',''),NEUTRINO_BENCH_DIR=str(dest),
        DEEP_MARKERS='1',NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
    env.update(specs[c]['env'])
    w=list(work)
    if specs[c]['mode']:
        w=[py,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null','--ablation-mode','files',
           '--tracedir',str(trace),'--python',str(root/'benchmarks/cache_python.py'),*w]
    kinds='write' if stack else 'read,write,writev,ioctl,futex,sched_yield,poll,ppoll,openat,close,mmap,munmap,brk,unlink,clock_nanosleep'
    cmd=[tool,'-f','-ttt','-T','-yy','-s','0','-e','trace='+kinds,'-o',str(raw)]
    if stack:cmd+=['-k']
    cmd+=w
    (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    print('LG64 SYSCALL START',name,flush=True)
    try:
        with (dest/'process.log').open('w') as fp:subprocess.run(cmd,cwd=root,env=env,stdout=fp,stderr=fp,timeout=600,check=True)
        w=json.loads((dest/'workload.json').read_text());assert w['output_digest']=='0be37a842462e3d71e853059945490c9fb052af64a530092d6b44dfac641298a'
        d=syscall_summary(raw);(dest/'syscalls.json').write_text(json.dumps(d,indent=2)+'\n')
        if stack:(dest/'stacks.json').write_text(json.dumps(stacks(raw),indent=2)+'\n')
        checks=[trace_check(p) for p in trace.glob('*/result/*.nts')]
        (dest/'complete.json').write_text(json.dumps(dict(passed=True,checks=checks,raw_retained=False))+'\n')
    finally:
        shutil.rmtree(trace)
        if raw.exists():raw.unlink()
    print('LG64 SYSCALL DONE',name,flush=True)
