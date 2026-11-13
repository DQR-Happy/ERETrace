"""Post-run identities and cleanup evidence for the LG64 residual diagnosis."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

primary=Path('/root/autodl-tmp/neutrino-lg64-detail-20261003')
follow=Path('/root/autodl-tmp/neutrino-lg64-stall-20261003')
assert (follow/'results/gpu-only-3round/complete.json').exists()
versions={'primary':primary,'fifo-owners':follow,
          'frozen-LG64':Path('/root/autodl-tmp/neutrino-ring-output-20261003')}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
sources={};binaries={}
for name,root in versions.items():
    paths=[]
    for folder in ['neutrino/src','neutrino/probe','neutrino/language','benchmarks','tests']:
        paths.extend(p for p in (root/folder).glob('*') if p.suffix in ['.py','.c','.h','.cu'])
    paths.extend(root/p for p in ['neutrino/cli.py','neutrino/build.py'])
    sources[name]={str(p.relative_to(root)):sha(p) for p in paths if p.exists()}
    binaries[name]={p:sha(root/p) for p in ['neutrino/build/libcuda.so.1','neutrino/build/preload.so',
        'neutrino/build/profile-preload.so','benchmarks/capture_pack.cubin','benchmarks/deep_cupti.so',
        'benchmarks/deep_cupti_activity.so','benchmarks/deep_cupti_gpuonly.so'] if (root/p).exists()}
model=Path('/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf')
paths=[model,Path('/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so'),
    Path('/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so.12'),
    Path('/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08')]
paths+=list(Path('/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin').glob('lib*.so'))
commands=[]
for argv in [['nvidia-smi','--query-gpu=name,memory.total,driver_version,compute_cap,pci.bus_id,clocks.current.sm,clocks.current.memory,temperature.gpu','--format=csv'],
    ['/usr/local/cuda/bin/nvcc','--version'],['uname','-a'],['cc','--version'],['stat','-f','-c','%T','/dev/shm']]:
    p=subprocess.run(argv,capture_output=True,text=True)
    commands.append(dict(argv=argv,returncode=p.returncode,output=p.stdout+p.stderr))
metadata={}
try:
    sys.path.insert(0,'/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/gguf-py')
    from gguf import GGUFReader
    r=GGUFReader(str(model))
    metadata['tensors']=[dict(name=t.name,shape=t.shape.tolist(),type=int(t.tensor_type),type_name=t.tensor_type.name)
        for t in r.tensors if t.name in ['output.weight','token_embd.weight']]
    metadata['vocab_size']=len(r.fields['tokenizer.ggml.tokens'].data)
except Exception as e:
    metadata['unavailable']=str(e)
d=dict(collected_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
    versions={k:str(v) for k,v in versions.items()},sources=sources,binaries=binaries,
    workload_and_libraries_sha256={str(p):sha(p) for p in paths},commands=commands,model_metadata=metadata,
    clocks_fixed=False,llama_source_revision='unverified: binary reports commit unknown',
    raw_traces_retained=False,tmpfs_remaining_task_directories=[str(p) for p in Path('/dev/shm/neutrino-lg64-detail').iterdir()],
    raw_syscall_remaining=[str(p) for p in Path('/dev/shm').glob('lg64-syscall-*')])
(primary/'results/provenance.json').write_text(json.dumps(d,indent=2)+'\n')
print('LG64 PROVENANCE COMPLETE')
