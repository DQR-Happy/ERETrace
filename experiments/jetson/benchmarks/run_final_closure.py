"""Portable paired experiment entry point; native, strong baselines and exact codec.

Every condition uses this checkout, the same corrected probe, model and adapter.
Only diagnostic runs load CUPTI or perform additional field restoration checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--backend', type=Path, required=True)
p.add_argument('--trace-root', type=Path, default=Path('/dev/shm/neutrino-final-closure'))
p.add_argument('--conditions', nargs='+', default=['N','G','R','VS','VP'])
p.add_argument('--rounds', type=int, default=7)
p.add_argument('--requests', type=int, default=3)
p.add_argument('--tokens', type=int, default=32)
p.add_argument('--repeat', type=int, default=1)
p.add_argument('--threads', type=int, default=48)
p.add_argument('--diagnostic', action='store_true')
p.add_argument('--gpu', action='store_true')
p.add_argument('--validate', action='store_true')
p.add_argument('--queries', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
base = dict(NEUTRINO_ARENA='1', NEUTRINO_CAPTURE_CLEAR='1',
    NEUTRINO_CAPTURE_ENCODER=str(root/'benchmarks/capture_pack.cubin'),
    NEUTRINO_ARENA_TOKEN_FLUSH='1', NEUTRINO_ARENA_HOST_MIB='32',
    NEUTRINO_ARENA_DEVICE_MIB='16', NEUTRINO_ARENA_INCREMENTAL='1',
    NEUTRINO_ARENA_BATCH_ENCODER='1', NEUTRINO_ARENA_DECOUPLE='0',
    NEUTRINO_ARENA_WRITER='1', NEUTRINO_ARENA_HOTPATH='3', NEUTRINO_CORE_MODE='shift',
    NEUTRINO_ARENA_CODEC_MIN_KIB='16')
reference = dict(base, NEUTRINO_ARENA_DELTA='1', NEUTRINO_ARENA_DELTA_WIDTH='3',
    NEUTRINO_ARENA_ELAPSED_WIDTH='2', NEUTRINO_ARENA_HIER='1',
    NEUTRINO_ARENA_REFERENCE='1', NEUTRINO_ARENA_LOCAL_BITS='12',
    NEUTRINO_ARENA_CODEC_MIN_KIB='16', NEUTRINO_ARENA_OMIT_ANCHOR='1')
specs = dict(N=dict(mode=None,env={}), S=dict(mode='files',env={'NEUTRINO_CORE_MODE':'shift'}),
             G=dict(mode='files',env=base), R=dict(mode='files',env=reference))
specs['RF'] = dict(mode='files',env=dict(reference,NEUTRINO_ARENA_FORCE_FALLBACK='1'))
specs['T'] = dict(mode='files',env=dict(reference,NEUTRINO_ARENA_TILE='4'))
specs['RB'] = dict(mode='files',env=dict(reference,NEUTRINO_ARENA_HOST_MIB='128',
                                     NEUTRINO_ARENA_DEVICE_MIB='64',NEUTRINO_ARENA_CHUNK_MIB='64'))
for code, lib in [('VS','serial'),('VP','parallel')]:
    specs[code] = dict(mode='files',env=dict(base, NEUTRINO_ARENA_NVCOMP='2',
        NEUTRINO_ARENA_CODEC_MIN_KIB='16', NEUTRINO_NVCOMP_CHUNK_KIB='64',
        NEUTRINO_NVPIPE_LIBRARY=str(root/f'benchmarks/nvcomp_pipeline_{lib}.so'),
        NEUTRINO_ARENA_HOST_MIB='128',NEUTRINO_ARENA_DEVICE_MIB='64',NEUTRINO_ARENA_CHUNK_MIB='64'))
if any(c not in specs for c in a.conditions): p.error('Unknown condition')
for c in a.conditions:
    e = specs[c]['env']
    e['NEUTRINO_BENCH_THREADS']=str(a.threads)
    if a.diagnostic or a.validate or a.queries: e.update(NEUTRINO_ARENA_DIAG='1',DEEP_SPACE='1')
    if a.validate and specs[c]['mode']: e.update(NEUTRINO_ARENA_VERIFY='1',NEUTRINO_CAPTURE_CLEAR='2')
    if a.queries and specs[c]['mode']: e.update(NEUTRINO_ARENA_VERIFY='1',NEUTRINO_QUERY_ORACLE='1')
    if a.gpu: e.update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',NEUTRINO_PROFILE_COMPAT='1',
        DEEP_CUPTI_CALLBACKS='0',DEEP_CUPTI_LIBRARY=str(root/'benchmarks/deep_cupti_gpuonly.so'))
a.output = a.output.resolve(); a.output.parent.mkdir(parents=True,exist_ok=True)
sp = a.output.with_name(a.output.name+'-specs.json')
if a.output.exists() or sp.exists(): raise RuntimeError('Refusing to overwrite an experiment')
sp.write_text(json.dumps({c:specs[c] for c in a.conditions},indent=2)+'\n')
command = [sys.executable,str(root/'benchmarks/run_arena_matrix.py'),
    '--output',str(a.output),'--trace-root',str(a.trace_root), '--spec',str(sp),
    '--rounds',str(a.rounds),'--conditions',*a.conditions,
    '--compiler-python',str(root/'benchmarks/cross_cache_python')]
if a.diagnostic or a.validate or a.queries: command += ['--diagnostic']
work = [sys.executable,str(root/'benchmarks/qwen_arena_workload.py'),
    '--adapter',str(root/'benchmarks/libqwen_dynamic.so'),'--backend',str(a.backend),
    '--model',str(a.model),'--warmup','2','--requests',str(a.requests),
    '--output-tokens',str(a.tokens),'--prompt-repeat',str(a.repeat)]
env = dict(os.environ, PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
           NEUTRINO_COMPILER_CACHE_ROOT=str(root/'compiler-cache'),NEUTRINO_BENCH_THREADS=str(a.threads))
if 'Llama-' in a.model.name: env['MECHANISM_PROMPT_STYLE']='llama3'
if a.queries: env['FINAL_QUERY_ANALYSIS']='1'
if a.validate: env['BULK_RECORD_VALIDITY']='1'
subprocess.run([*command,'--',*work],cwd=root,env=env,check=True)
h = hashlib.sha256()
with a.model.open('rb') as f:
    for b in iter(lambda:f.read(8<<20),b''): h.update(b)
(a.output/'inputs.json').write_text(json.dumps(dict(model=str(a.model),model_sha256=h.hexdigest(),
    backend=str(a.backend),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}),indent=2)+'\n')
