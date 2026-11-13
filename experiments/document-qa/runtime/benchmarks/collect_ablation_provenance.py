"""Record actual artifacts, never infer a source revision from an old report."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

roots=[Path('/root/autodl-tmp/neutrino-ablation-20261002'),
       Path('/root/autodl-tmp/neutrino-ablation-protocol-20261003')]
relative=['neutrino/src/cuda.c','neutrino/src/ablation.h','neutrino/src/timing.h',
          'neutrino/src/common.h','neutrino/cli.py','neutrino/probe/cuda.py',
          'neutrino/build/libcuda.so.1','neutrino/build/preload.so',
          'neutrino/build/profile-preload.so']
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda:fp.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def command(args):
    p=subprocess.run(args,text=True,capture_output=True)
    return dict(argv=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
sources={str(root):{r:sha(root/r) for r in relative if (root/r).exists()} for root in roots}
external=[Path('/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so'),
          Path('/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf')]
external+=list(Path('/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin').glob('*.so*'))
data=dict(sources=sources,artifacts={str(p):sha(p) for p in external if p.is_file()},
          page_size=os.sysconf('SC_PAGE_SIZE'),
          commands=[command(a) for a in [['nvidia-smi','-q'],['lscpu'],['uname','-a'],
            ['/usr/local/cuda/bin/nvcc','--version'],
            ['/root/autodl-tmp/neutrino-profile-tools/nsys/opt/nvidia/nsight-systems/2024.2.3/bin/nsys','--version']]],
          llama_revision='unknown; no verified binary source revision',
          primary_source_manifest='source-main/manifest.json',
          note='Current primary CLI includes later profiling-only loader branch; main run frozen hashes are in primary manifest.')
print(json.dumps(data,indent=2))
