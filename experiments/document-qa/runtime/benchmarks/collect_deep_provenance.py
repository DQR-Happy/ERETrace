"""Freeze measured code and build/tool/library identities without raw traces."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

root=Path(__file__).resolve().parents[1];out=root/'results/deep'
bases=[Path('/root/autodl-tmp')/('neutrino-'+x+'-20261003') for x in ['deep','deep-async','deep-stream']]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
sources={};binaries={}
for base in bases:
    source={}
    for folder in ['neutrino/src','neutrino/probe','benchmarks']:
        for p in (base/folder).glob('*'):
            if p.suffix in ('.c','.h','.py','.cu'):source[str(p.relative_to(base))]=sha(p)
    for p in [base/'neutrino/cli.py',base/'neutrino/build.py']:source[str(p.relative_to(base))]=sha(p)
    sources[str(base)]=source
    binaries[str(base)]={str(p.relative_to(base)):sha(p) for p in [base/'neutrino/build/libcuda.so.1',base/'neutrino/build/preload.so',base/'benchmarks/deep_cupti.so'] if p.exists()}
paths=[Path('/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf'),
 Path('/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so'),
 Path('/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so.12')]
paths+=list(Path('/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin').glob('lib*.so'))
commands=[['nvidia-smi','--query-gpu=name,memory.total,driver_version,compute_cap,pci.bus_id','--format=csv'],
 ['/usr/local/cuda/bin/nvcc','--version'],['cc','--version'],['stat','-f','-c','%T','/dev/shm']]
results=[]
for command in commands:
    p=subprocess.run(command,capture_output=True,text=True);results.append(dict(argv=command,returncode=p.returncode,output=p.stdout+p.stderr))
data=dict(collected_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
 sources=sources,binaries=binaries,workload_and_libraries_sha256={str(p):sha(p) for p in paths},commands=results,
 tools=json.loads(Path('/root/autodl-tmp/neutrino-deep-tools-20261003/setup.json').read_text()),
 swaps=Path('/proc/swaps').read_text(),numa_gpu_node=Path('/sys/bus/pci/devices/0000:1c:00.0/numa_node').read_text(),
 cpu_node_lists={n:Path(f'/sys/devices/system/node/node{n}/cpulist').read_text() for n in (0,1)},
 llama_source_revision='unverified: prior binary reports commit unknown',
 copy_correlation_schema=2,raw_traces_retained=False)
(out/'provenance.json').write_text(json.dumps(data,indent=2)+'\n')
print('PROVENANCE COMPLETE',flush=True)
