"""Clean, path-configurable build for the final fixed-warp collection study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--llama', type=Path, required=True)
p.add_argument('--nvcomp', type=Path, required=True)
p.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
p.add_argument('--arch', default='sm_86')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
env = dict(os.environ, PATH=str(a.cuda/'bin')+':'+os.environ.get('PATH', ''))
commands = []
def run(cmd):
    cmd = list(map(str, cmd)); commands.append(cmd)
    print('BUILD', cmd, flush=True)
    subprocess.run(cmd, cwd=root, env=env, check=True)
run([sys.executable, root/'benchmarks/build_cross_device.py', '--llama', a.llama, '--arch', a.arch])
for name, flags in [('serial', ['-DNVPIPE_SERIAL_LAYOUT']), ('parallel', [])]:
    run([a.cuda/'bin/nvcc', '-O3', '-std=c++17', '-arch='+a.arch, '-shared',
         '-Xcompiler', '-fPIC', *flags, root/'benchmarks/nvcomp_pipeline.cu',
         '-I'+str(a.nvcomp/'include'), '-L'+str(a.nvcomp/'lib64'),
         '-Xlinker', '-rpath', '-Xlinker', a.nvcomp/'lib64', '-l:libnvcomp.so.5',
         '-o', root/f'benchmarks/nvcomp_pipeline_{name}.so'])
run(['gcc', '-O2', '-shared', '-fPIC', root/'benchmarks/deep_cupti_gpuonly.c',
     '-I'+str(a.cuda/'include'), '-I'+str(a.cuda/'extras/CUPTI/include'),
     '-L'+str(a.cuda/'lib64'), '-L'+str(a.cuda/'extras/CUPTI/lib64'),
     '-Wl,-rpath,'+str(a.cuda/'lib64'), '-Wl,-rpath,'+str(a.cuda/'extras/CUPTI/lib64'),
     '-lcupti', '-ldl', '-lpthread', '-o', root/'benchmarks/deep_cupti_gpuonly.so'])
preload = root/'benchmarks/profile_preload.c'
run(['gcc', '-O2', '-shared', '-fPIC', preload, '-ldl', '-o', root/'neutrino/build/profile-preload.so'])
assets = [root/'benchmarks/capture_pack.cubin', root/'benchmarks/libqwen_dynamic.so',
          *root.glob('benchmarks/nvcomp_pipeline_*.so'), root/'benchmarks/deep_cupti_gpuonly.so',
          *root.glob('neutrino/build/*.so'), a.nvcomp/'lib64/libnvcomp.so.5',
          *a.llama.glob('build/bin/lib*.so*')]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()
data = dict(commands=commands, cuda=str(a.cuda), nvcomp=str(a.nvcomp), arch=a.arch,
            hashes={str(q):sha(q) for q in assets if q.is_file()},
            python=sys.version, nvcc=subprocess.check_output([str(a.cuda/'bin/nvcc'),'--version'], text=True),
            smi=subprocess.check_output(['nvidia-smi','-q'], text=True))
(root/'final-build.json').write_text(json.dumps(data, indent=2)+'\n')
