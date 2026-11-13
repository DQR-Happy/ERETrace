#!/usr/bin/env python3
"""Build CUDA components explicitly; pip installation never compiles a driver hook."""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def run(cmd):
    print('+', ' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def resolve_driver(explicit):
    if explicit:
        p=Path(explicit).resolve(strict=True)
        if 'stubs' in p.parts:raise ValueError('Use the installed NVIDIA driver, not toolkit stubs')
        return p
    loader=shutil.which('ldconfig') or '/sbin/ldconfig'
    out=subprocess.check_output([loader,'-p'],text=True)
    for line in out.splitlines():
        if 'libcuda.so.1 ' in line and '=>' in line:
            p=Path(line.split('=>')[1].strip()).resolve()
            if p.exists() and 'stubs' not in p.parts and 'i386' not in str(p):return p
    raise RuntimeError('Driver not found: pass --driver /absolute/path/to/libcuda.so.1')
def main():
    global ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',type=Path,default=ROOT,help='Explicit alternate runtime, e.g. the document-QA snapshot')
    p.add_argument('--cuda-home',type=Path,default=Path(os.getenv('CUDA_HOME','/usr/local/cuda')))
    p.add_argument('--driver');p.add_argument('--arch',default='86',help='86 (3090), 87 (Orin), 90 (H800)')
    p.add_argument('--cc',default='cc');p.add_argument('--cxx',default='c++')
    p.add_argument('--llama-source',type=Path);p.add_argument('--llama-build',type=Path)
    p.add_argument('--nvcomp-root',type=Path,help='Optional SDK prefix containing include/ and lib/ or lib64/')
    a=p.parse_args()
    ROOT=a.runtime_root.resolve()
    if platform.system()!='Linux':p.error('CUDA capture builds require Linux; CPU tests and result summaries work on macOS')
    if not a.arch.isdigit():p.error('--arch must be a numeric compute capability, e.g. 86')
    if bool(a.llama_source)!=bool(a.llama_build):p.error('Set both --llama-source and --llama-build')
    cuda=a.cuda_home.resolve();nvcc=cuda/'bin/nvcc';header=cuda/'include/cuda.h'
    if not header.exists() or not nvcc.exists():p.error('CUDA toolkit headers/compiler missing')
    driver=resolve_driver(a.driver);build=ROOT/'neutrino/build';build.mkdir(exist_ok=True)
    out=ROOT/'build';out.mkdir(exist_ok=True)
    import toml
    config={'system':{'NEUTRINO_MODE':'CUDA','NEUTRINO_DRIVER_HEADER_NAME':'cuda.h','NEUTRINO_DRIVER_HEADER_DIR':str(header.parent),'NEUTRINO_HOOK_DRIVER_LIB_NAME':'libcuda.so.1','NEUTRINO_REAL_DRIVER_LIB_NAME':driver.name,'NEUTRINO_REAL_DRIVER_LIB_DIR':str(driver.parent)}}
    run([sys.executable,ROOT/'neutrino/src/parse.py',header,driver])
    run([a.cc,ROOT/'neutrino/src/cuda.c','-fPIC','-shared','-O3','-I',header.parent,'-o',build/'libcuda.so.1','-ldl','-lpthread'])
    run([a.cc,ROOT/'neutrino/src/preload.c','-fPIC','-shared','-O3','-o',build/'preload.so','-ldl'])
    link=build/'libcuda.so'
    if link.is_symlink():link.unlink()
    elif link.exists():raise RuntimeError(f'Refusing to overwrite regular file {link}')
    link.symlink_to('libcuda.so.1')
    (build/'config.toml').write_text(toml.dumps(config))
    run([nvcc,'-std=c++17','-O3','-arch=sm_'+a.arch,'--cubin',ROOT/'benchmarks/capture_pack.cu','-o',out/'capture_pack.cubin'])
    if a.llama_source:
        src=a.llama_source.resolve();lb=a.llama_build.resolve()/'bin'
        if not (lb/'libllama.so').exists():raise FileNotFoundError(lb/'libllama.so')
        run([a.cxx,'-std=c++17','-O3','-fPIC','-shared',ROOT/'benchmarks/llama_adapter_extended.cpp','-I',src/'include','-I',src/'ggml/include','-L',lb,'-Wl,-rpath,'+str(lb),'-lllama','-lggml','-lggml-base','-o',out/'libllama_adapter.so'])
    if a.llama_source:
        run([a.cxx,'-std=c++17','-O3','-fPIC','-shared',ROOT/'experiments/b16/llama_adapter_batch16.cpp','-I',src/'include','-I',src/'ggml/include','-L',lb,'-Wl,-rpath,'+str(lb),'-lllama','-lggml','-lggml-base','-o',out/'libllama_adapter_b16.so'])
    if a.nvcomp_root:
        sdk=a.nvcomp_root.resolve();lib=next((sdk/s for s in ('lib','lib64') if (sdk/s/'libnvcomp.so').exists()),None)
        if lib is None:raise FileNotFoundError('nvCOMP shared library missing under lib/ or lib64/')
        run([nvcc,'-std=c++17','-O3','-arch=sm_'+a.arch,'-shared','-Xcompiler=-fPIC',ROOT/'benchmarks/nvcomp_pipeline.cu','-I',sdk/'include','-L',lib,'-Xlinker=-rpath','-Xlinker='+str(lib),'-lnvcomp','-o',out/'nvcomp_pipeline.so'])
    manifest={'platform':platform.platform(),'machine':platform.machine(),'arch':a.arch,'cuda_version':subprocess.check_output([nvcc,'--version'],text=True),'driver_library':str(driver),'driver_sha256':sha(driver),'artifacts':{str(f.relative_to(ROOT)):sha(f) for f in list(build.glob('*.so*'))+list(out.glob('*.so'))+list(out.glob('*.cubin')) if f.is_file()}}
    (out/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built successfully. Manifest: build/build-manifest.json')
if __name__=='__main__':main()
