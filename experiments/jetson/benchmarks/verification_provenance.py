"""Compact reproducibility metadata; never copies a model, library or raw trace."""
import hashlib,json,platform,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def command(c):
    r=subprocess.run(c,capture_output=True,text=True);return dict(command=c,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
files=list((root/'neutrino/src').glob('*'))+list((root/'benchmarks').glob('*.py'))+list((root/'benchmarks').glob('*.cu'))+list((root/'benchmarks').glob('*.cpp'))+list((root/'benchmarks').glob('*.c'))
files+=list((root/'tests').glob('*.py'))+list((root/'benchmarks').glob('*.cuh'))+[root/'benchmarks/reference_shape_workload',root/'benchmarks/hotpath_shape_workload',root/'benchmarks/nvcomp_pipeline.so']
for previous in ['neutrino-reference-cutoff-20261004','neutrino-codec-comparison-20261004','neutrino-nvcomp-pipeline-20261004','neutrino-hotpath-20261004']:
    p=Path('/root/autodl-tmp')/previous
    files += [p/'neutrino/build/libcuda.so.1',p/'benchmarks/capture_pack.cubin',p/'neutrino/src/arena_capture.h',p/'neutrino/src/cuda.c',p/'benchmarks/nvcomp_pipeline.so']
vendor=Path('/root/autodl-tmp/neutrino-mechanism-20261004/vendor/nvidia/libnvcomp')
files += [vendor/'lib64/libnvcomp.so.5',vendor/'include/nvcomp/cascaded.h',vendor/'include/nvcomp.h']
if root.name.startswith('neutrino-reference'):
    previous=Path('/root/autodl-tmp/neutrino-hierarchy-20261004')
    files += [previous/'neutrino/build/libcuda.so.1',previous/'benchmarks/capture_pack.cubin',previous/'neutrino/src/arena_capture.h']
    if root.name!='neutrino-reference-20261004':
        previous=Path('/root/autodl-tmp/neutrino-reference-20261004')
        files += [previous/'neutrino/build/libcuda.so.1',previous/'benchmarks/capture_pack.cubin',previous/'neutrino/src/arena_capture.h']
files+=[root/'neutrino/build/libcuda.so.1',root/'neutrino/build/preload.so',root/'neutrino/build/profile-preload.so',root/'benchmarks/capture_pack.cubin',root/'benchmarks/libqwen_dynamic.so']
files += [Path('/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf'),Path('/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so')]
files+=list(Path('/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin').glob('lib*.so'))
for cfg in (root/'results').glob('*/config.json'):
    d=json.loads(cfg.read_text());cmd=d.get('command',[])
    if '--model' in cmd:files.append(Path(cmd[cmd.index('--model')+1]))
files += [Path('/root/autodl-tmp/neutrino-arena-20261003/benchmarks/deep_cupti_gpuonly.so'),Path('/root/autodl-tmp/neutrino-arena-20261003/benchmarks/deep_cupti_gpuonly.c')]
files=list(dict.fromkeys(files))
data=dict(root=str(root),python=sys.version,platform=platform.platform(),sha256={str(p):sha(p) for p in files if p.is_file()},commands=[command(c) for c in [['nvidia-smi','-q'],['/usr/local/cuda/bin/nvcc','--version'],['uname','-a'],['df','-h','/dev/shm'],['git','-C','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp','rev-parse','HEAD']]])
(root/'results/provenance.json').write_text(json.dumps(data,indent=2)+'\n')
