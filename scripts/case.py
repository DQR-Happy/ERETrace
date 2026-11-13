#!/usr/bin/env python3
"""Build/run the complete document-QA case using checkout-relative paths."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'experiments/document-qa'
RUNTIME=CASE/'runtime'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='action',required=True)
    b=sp.add_parser('build');b.add_argument('--llama-source',type=Path,required=True);b.add_argument('--llama-build',type=Path,required=True);b.add_argument('--arch',default='86');b.add_argument('--cuda-home',default='/usr/local/cuda');b.add_argument('--driver');b.add_argument('--nvcomp-root',type=Path);b.add_argument('--dry-run',action='store_true')
    r=sp.add_parser('run');r.add_argument('--model',type=Path,required=True);r.add_argument('--backend',type=Path,required=True);r.add_argument('--condition',choices=['native','serial','async','nvcomp','ours'],default='ours');r.add_argument('--output',type=Path,required=True);r.add_argument('--prompt-index',type=int,choices=[0,1,2],default=0);r.add_argument('--warmup',type=int,default=2);r.add_argument('--requests',type=int,default=1);r.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.action=='build':
        src=a.llama_source.resolve();lib=a.llama_build.resolve()/'bin'
        commands=[[sys.executable,str(ROOT/'scripts/build.py'),'--runtime-root',str(RUNTIME),'--arch',a.arch,'--cuda-home',a.cuda_home]]
        if a.driver:commands[0]+= ['--driver',a.driver]
        if a.nvcomp_root:commands[0]+= ['--nvcomp-root',str(a.nvcomp_root.resolve())]
        commands.append(['c++','-O3','-std=c++17','-fPIC','-shared',str(RUNTIME/'benchmarks/llama_adapter_dynamic.cpp'),'-I'+str(src/'include'),'-I'+str(src/'ggml/include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml','-lggml-base','-o',str(RUNTIME/'benchmarks/libcase.so')])
        if a.dry_run:print(json.dumps(commands,indent=2));return
        for cmd in commands:subprocess.run(cmd,check=True,cwd=ROOT)
        return
    if a.warmup<0 or a.requests<1:p.error('Positive request count and nonnegative warmup required')
    out=a.output.resolve();trace=out/'trace';backend=a.backend.resolve();model=a.model.resolve()
    env={k:v for k,v in os.environ.items() if not k.startswith(('NEUTRINO_','CASE_'))}
    env.update(PYTHONPATH=str(RUNTIME)+os.pathsep+str(CASE/'code'),CASE_BACKEND=str(backend),CASE_MODEL=str(model),CASE_BATCH='512',CASE_ANALYZE='0',CASE_DISCARD_WARMUP='1',CASE_PROMPT_INDEX=str(a.prompt_index),GGML_CUDA_DISABLE_GRAPHS='1',NEUTRINO_BENCH_THREADS='8',NEUTRINO_CASE_SIDECAR=str(out/'launches.tsv'),NEUTRINO_CAPTURE_CLEAR='1',NEUTRINO_CORE_MODE='shift',NEUTRINO_ABLATION_STATS='1',NEUTRINO_NVPIPE_LIBRARY=str(RUNTIME/'build/nvcomp_pipeline.so'))
    env['LD_LIBRARY_PATH']=str(backend)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    workload=[sys.executable,str(CASE/'code/case_workload.py'),'--root',str(out),'--out',str(out),'--prompts',str(CASE/'code/prompts.json'),'--warmup',str(a.warmup),'--requests',str(a.requests),'--set','heldout']
    command=workload
    if a.condition!='native':
        command=[sys.executable,'-m','neutrino.cli','-p','block_sched','--callback','/dev/null','--python',sys.executable,'--ablation-mode','files','--tracedir',str(trace),*workload]
        if a.condition=='serial':env.update(NEUTRINO_ABLATION_WRITER='1',NEUTRINO_ABLATION_FILE_BUFFER='1048576',NEUTRINO_CAPTURE_CLEAR='0')
        else:
            env.update(NEUTRINO_ARENA='1',NEUTRINO_ARENA_HOST_MIB='128',NEUTRINO_ARENA_DEVICE_MIB='64',NEUTRINO_ARENA_CHUNK_MIB='64',NEUTRINO_ARENA_INCREMENTAL='1',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',NEUTRINO_ARENA_TOKEN_FLUSH='1',NEUTRINO_ARENA_WRITER='1',NEUTRINO_ARENA_HOTPATH='3',NEUTRINO_ARENA_CODEC_MIN_KIB='16',NEUTRINO_CAPTURE_ENCODER=str(RUNTIME/'build/capture_pack.cubin'))
            if a.condition=='async':env['NEUTRINO_CASE_RAW']='1'
            if a.condition=='nvcomp':env.update(NEUTRINO_ARENA_NVCOMP='2',NEUTRINO_NVCOMP_CHUNK_KIB='64')
            if a.condition=='ours':env.update(NEUTRINO_ARENA_DELTA='1',NEUTRINO_ARENA_DELTA_WIDTH='3',NEUTRINO_ARENA_ELAPSED_WIDTH='2',NEUTRINO_ARENA_HIER='1',NEUTRINO_ARENA_REFERENCE='1',NEUTRINO_ARENA_LOCAL_BITS='12',NEUTRINO_ARENA_OMIT_ANCHOR='1')
    config={'command':command,'cwd':str(RUNTIME),'environment':{k:v for k,v in env.items() if k.startswith(('CASE_','NEUTRINO_','GGML_'))},'condition':a.condition,'endpoint':'request start through complete trace file close; analysis disabled'}
    if a.dry_run:print(json.dumps(config,indent=2));return
    if out.exists():p.error('Use a new output directory')
    for f in [model,backend/'libllama.so',RUNTIME/'benchmarks/libcase.so']:
        if not f.is_file():p.error('Missing '+str(f))
    if a.condition=='nvcomp' and not (RUNTIME/'build/nvcomp_pipeline.so').is_file():p.error('Build the optional nvCOMP case adapter first')
    out.mkdir(parents=True);(out/'source').symlink_to(RUNTIME,target_is_directory=True);trace.mkdir()
    config.update(model_sha256=digest(model),adapter_sha256=digest(RUNTIME/'benchmarks/libcase.so'))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with (out/'run.log').open('w') as log:subprocess.run(command,cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print((out/'workload.json').read_text())
if __name__=='__main__':main()
