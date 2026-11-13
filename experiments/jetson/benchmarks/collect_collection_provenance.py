"""Post-measurement build/library identities; no raw trace retention."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
versions={
    'frozen-T8':Path('/root/autodl-tmp/neutrino-deep-stream-20261003'),
    'source':Path('/root/autodl-tmp/neutrino-bulk-20261003'),
    'source-ring':Path('/root/autodl-tmp/neutrino-ring-output-20261003'),
    'source-fixed':Path('/root/autodl-tmp/neutrino-collection-fixed-20261003'),
}
flat={
    'source':['bulk_capture.h','capture_codec.h','cuda.c','ablation.h','common.h','timing.h','async_capture.h'],
    'source-ring':['bulk_capture.h','capture_codec.h','legacy_capture.h','async_capture.h','cuda.c','ablation.h'],
    'source-fixed':['bulk_capture.h','capture_codec.h','legacy_capture.h','async_capture.h','cuda.c','ablation.h','common.h','timing.h'],
}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for data in iter(lambda:fp.read(1048576),b''):h.update(data)
    return h.hexdigest()
sources={};binaries={};freeze={}
for label,base in versions.items():
    files=[]
    for folder in ('neutrino/src','neutrino/probe','neutrino/language','benchmarks','tests'):
        files.extend(p for p in (base/folder).glob('*') if p.suffix in ('.h','.c','.cu','.py'))
    files.extend(base/p for p in ('neutrino/cli.py','neutrino/build.py'))
    sources[label]={str(p.relative_to(base)):sha(p) for p in files if p.exists()}
    binaries[label]={p:sha(base/p) for p in ('neutrino/build/libcuda.so.1','neutrino/build/preload.so','neutrino/build/profile-preload.so',
        'benchmarks/capture_pack.cubin','benchmarks/deep_cupti.so','benchmarks/clock_domains') if (base/p).exists()}
    if label in flat:
        freeze[label]={p:sha(base/'neutrino/src'/p) for p in flat[label]}
        if label=='source-ring':
            freeze[label].update({p:sha(base/'benchmarks'/p) for p in ('capture_pack.cu','clock_domains.cu')})
paths=[Path('/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf'),
    Path('/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so'),
    Path('/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so.12'),Path('/usr/lib/x86_64-linux-gnu/libzstd.so.1')]
paths+=list(Path('/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin').glob('lib*.so'))
commands=[['nvidia-smi','--query-gpu=name,memory.total,driver_version,compute_cap,pci.bus_id','--format=csv'],
    ['/usr/local/cuda/bin/nvcc','--version'],['cc','--version'],['stat','-f','-c','%T','/dev/shm']]
outputs=[]
for command in commands:
    p=subprocess.run(command,capture_output=True,text=True)
    outputs.append(dict(argv=command,returncode=p.returncode,output=p.stdout+p.stderr))
data=dict(collected_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
    versions={k:str(v) for k,v in versions.items()},sources=sources,source_hashes=freeze,binaries=binaries,
    workload_and_libraries_sha256={str(p):sha(p) for p in paths},commands=outputs,
    clocks_fixed=False,llama_source_revision='unverified: binary reports commit unknown',
    copy_correlation_schema=2,raw_traces_retained=False,
    tmpfs_remaining_task_directories=[str(p) for p in Path('/dev/shm/neutrino-ablation').iterdir()],
    swaps=Path('/proc/swaps').read_text())
out=root/'results/provenance.json';out.write_text(json.dumps(data,indent=2)+'\n')
print('COLLECTION PROVENANCE COMPLETE',flush=True)
