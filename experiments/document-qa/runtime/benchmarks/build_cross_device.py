"""Build this isolated checkout against the device's existing llama/CUDA installation."""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--llama', type=Path, required=True)
p.add_argument('--arch', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
env = dict(os.environ, PATH='/usr/local/cuda/bin:' + os.environ.get('PATH', ''))
def run(cmd):
    print(cmd, flush=True)
    subprocess.run(list(map(str, cmd)), cwd=root, env=env, check=True)
run([sys.executable, root/'neutrino/build.py'])
shutil.copyfile(root/'neutrino/probe/cuda.py', root/'neutrino/build/process.py')
run(['nvcc', '-O3', '-std=c++17', '-arch='+a.arch, '-cubin', root/'benchmarks/capture_pack.cu', '-o', root/'benchmarks/capture_pack.cubin'])
run(['g++', '-O3', '-shared', '-fPIC', root/'benchmarks/llama_adapter_dynamic.cpp', '-I'+str(a.llama/'include'), '-I'+str(a.llama/'ggml/include'), '-L'+str(a.llama/'build/bin'), '-Wl,-rpath,'+str(a.llama/'build/bin'), '-lllama', '-lggml', '-lggml-base', '-o', root/'benchmarks/libqwen_dynamic.so'])
launcher = root/'benchmarks/cross_cache_python'
source = (root/'benchmarks/cache_python.py').read_text().split('\n', 1)[1]
launcher.write_text('#!'+sys.executable+'\n'+source)
launcher.chmod(0o755)
cache = root/'compiler-cache'
cache.mkdir(exist_ok=True)
(cache/'hardware.json').write_text(json.dumps(dict(arch=a.arch, platform=platform.platform(), smi=subprocess.check_output(['nvidia-smi','-q'],text=True)))+'\n')
print('CROSS BUILD COMPLETE', flush=True)
