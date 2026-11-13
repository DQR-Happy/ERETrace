"""Same-build matched capture controls on an existing second GPU. No raw archive."""
import argparse, json, os, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--backend',required=True)
p.add_argument('--model',required=True)
p.add_argument('--conditions',nargs='+',default=['N','I','K16','E','EAH'])
p.add_argument('--rounds',type=int,default=5)
p.add_argument('--requests',type=int,default=1)
p.add_argument('--threads',type=int,default=6)
p.add_argument('--repeat',type=int,default=1)
p.add_argument('--diagnostic',action='store_true')
p.add_argument('--validate',action='store_true')
p.add_argument('--gpu',action='store_true')
p.add_argument('--discard-warmup-traces',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True)
base=dict(NEUTRINO_ARENA='1',NEUTRINO_CAPTURE_CLEAR='1',NEUTRINO_CAPTURE_ENCODER=str(root/'benchmarks/capture_pack.cubin'),NEUTRINO_ARENA_TOKEN_FLUSH='1',NEUTRINO_ARENA_HOST_MIB='32',NEUTRINO_ARENA_DEVICE_MIB='16',NEUTRINO_ARENA_INCREMENTAL='1')
k={**base,'NEUTRINO_ARENA_BATCH_ENCODER':'1','NEUTRINO_ARENA_DECOUPLE':'0','NEUTRINO_ARENA_DELTA':'1','NEUTRINO_ARENA_WRITER':'1','NEUTRINO_ARENA_DELTA_WIDTH':'3','NEUTRINO_ARENA_ELAPSED_WIDTH':'2','NEUTRINO_ARENA_HIER':'1','NEUTRINO_ARENA_REFERENCE':'1','NEUTRINO_ARENA_LOCAL_BITS':'12','NEUTRINO_ARENA_CODEC_MIN_KIB':'16'}
spec={'N':dict(mode=None,env={}), 'I':dict(mode='files',env=base), 'K16':dict(mode='files',env=k), 'E':dict(mode='files',env={**k,'NEUTRINO_ARENA_OMIT_ANCHOR':'1'}), 'EAH':dict(mode='files',env={**k,'NEUTRINO_ARENA_OMIT_ANCHOR':'1','NEUTRINO_ARENA_HOTPATH':'3'})}
spec['B13']=dict(mode='files',env={**base,'NEUTRINO_ARENA_BATCH_ENCODER':'1','NEUTRINO_ARENA_DECOUPLE':'0','NEUTRINO_ARENA_WRITER':'1','NEUTRINO_ARENA_CODEC_MIN_KIB':'16'})
spec['EFAST']=dict(mode='files',env={**spec['EAH']['env'],'NEUTRINO_CAPTURE_ENCODER':str(root/'benchmarks/capture_pack_fast.cubin')})
spec['EOLD']=dict(mode='files',env={**spec['EAH']['env'],'NEUTRINO_CAPTURE_ENCODER':str(root.parent/'memory-control/benchmarks/capture_pack.cubin')})
for c in a.conditions:
    spec[c]['env']['NEUTRINO_BENCH_THREADS']=str(a.threads)
    spec[c]['env']['NEUTRINO_CROSS_MEMORY']='1'
    if a.discard_warmup_traces:spec[c]['env']['NEUTRINO_BENCH_DISCARD_WARMUP']='1'
    if a.diagnostic:spec[c]['env'].update(NEUTRINO_ARENA_DIAG='1',DEEP_SPACE='1')
    if a.validate and c!='N':spec[c]['env'].update(NEUTRINO_ARENA_VERIFY='1',NEUTRINO_CAPTURE_CLEAR='2',NEUTRINO_ARENA_DIAG='1')
    if a.gpu:spec[c]['env'].update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',DEEP_CUPTI_CALLBACKS='0',NEUTRINO_PROFILE_COMPAT='1',DEEP_CUPTI_LIBRARY=str(root/'benchmarks/deep_cupti_gpuonly.so'))
sp=out/(a.name+'-specs.json');sp.write_text(json.dumps(spec,indent=2)+'\n')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),NEUTRINO_COMPILER_CACHE_ROOT=str(root/'compiler-cache'))
if 'Llama-3.2' in a.model:env['MECHANISM_PROMPT_STYLE']='llama3'
cmd=[sys.executable,str(root/'benchmarks/run_arena_matrix.py'),'--output',str(out/a.name),'--trace-root','/dev/shm/neutrino-cross-device-20261004','--spec',str(sp),'--conditions',*a.conditions,'--rounds',str(a.rounds),'--timeout','1200','--compiler-python',str(root/'benchmarks/cross_cache_python')]
if a.diagnostic or a.validate:cmd+=['--diagnostic']
cmd+=['--',sys.executable,str(root/'benchmarks/qwen_arena_workload.py'),'--adapter',str(root/'benchmarks/libqwen_dynamic.so'),'--backend',a.backend,'--model',a.model,'--warmup','2','--requests',str(a.requests),'--prompt-repeat',str(a.repeat)]
subprocess.run(cmd,cwd=root,env=env,check=True)
