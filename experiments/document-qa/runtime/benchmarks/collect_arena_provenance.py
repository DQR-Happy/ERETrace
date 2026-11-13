"""Capture actual source/library identities after both isolated suites finish."""
import hashlib,json,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
primary=Path('/root/autodl-tmp/neutrino-arena-20261003');follow=Path('/root/autodl-tmp/neutrino-arena-follow-20261003')
versions={'primary':primary,'generic':follow,'frozen-best':Path('/root/autodl-tmp/neutrino-lg64-stall-20261003')}
assert all((p/'results/suite-complete.json').exists() for p in [primary,follow])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
sources={};binaries={}
for name,root in versions.items():
    paths=[]
    for directory in ['neutrino/src','neutrino/probe','neutrino/language','benchmarks','tests']:
        paths.extend(p for p in (root/directory).glob('*') if p.suffix in ['.py','.c','.h','.cu'])
    paths.extend(root/p for p in ['neutrino/cli.py','neutrino/build.py'])
    sources[name]={str(p.relative_to(root)):sha(p) for p in paths}
    binaries[name]={p:sha(root/p) for p in ['neutrino/build/libcuda.so.1','neutrino/build/preload.so','neutrino/build/profile-preload.so','benchmarks/capture_pack.cubin','benchmarks/deep_cupti_gpuonly.so'] if (root/p).exists()}
model=Path('/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf')
paths=[model,Path('/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so'),Path('/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08'),Path('/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so.12')]
paths+=list(Path('/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin').glob('lib*.so'))
commands=[]
for cmd in [['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version,compute_cap,clocks.current.sm,clocks.current.memory,temperature.gpu','--format=csv'],['/usr/local/cuda/bin/nvcc','--version'],['uname','-a'],['cc','--version'],['stat','-f','-c','%T','/dev/shm']]:
    r=subprocess.run(cmd,capture_output=True,text=True);commands.append(dict(argv=cmd,returncode=r.returncode,output=r.stdout+r.stderr))
leftovers={str(p):[str(x) for x in p.rglob('*')] for p in [Path('/dev/shm/neutrino-arena-20261003'),Path('/dev/shm/neutrino-arena-follow')]}
result=dict(collected_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),versions={k:str(v) for k,v in versions.items()},sources=sources,binaries=binaries,workload_and_libraries_sha256={str(p):sha(p) for p in paths},commands=commands,clocks_fixed=False,llama_source_revision='unverified; binary reports commit unknown',raw_traces_retained=False,tmpfs_leftovers=leftovers)
assert not any(leftovers.values())
(follow/'results/provenance.json').write_text(json.dumps(result,indent=2)+'\n');print('Provenance and cleanup verified')
