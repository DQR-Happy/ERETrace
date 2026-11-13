"""Independent runs for frozen per-function policies; no online model replay."""
import argparse,json,os,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--rounds',type=int,default=7)
p.add_argument('--requests',type=int,default=3)
p.add_argument('--conditions',nargs='+',default=['N','S','A','R','T'])
p.add_argument('--gpu13',action='store_true')
p.add_argument('--gpu',action='store_true')
p.add_argument('--diagnostic',action='store_true')
p.add_argument('--validate',action='store_true')
p.add_argument('--prompt-repeat',type=int,default=1)
p.add_argument('--output-tokens',type=int,default=32)
p.add_argument('--model',default='/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf')
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True)
base=dict(NEUTRINO_ARENA='1',NEUTRINO_CAPTURE_CLEAR='1',NEUTRINO_CAPTURE_ENCODER=str(root/'benchmarks/capture_pack.cubin'),NEUTRINO_ARENA_TOKEN_FLUSH='1',NEUTRINO_ARENA_HOST_MIB='32',NEUTRINO_ARENA_DEVICE_MIB='16',NEUTRINO_ARENA_INCREMENTAL='1',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',NEUTRINO_ARENA_DELTA='1',NEUTRINO_ARENA_WRITER='1',NEUTRINO_ARENA_DELTA_WIDTH='3',NEUTRINO_ARENA_ELAPSED_WIDTH='2',NEUTRINO_ARENA_HIER='1',NEUTRINO_ARENA_REFERENCE='1',NEUTRINO_ARENA_LOCAL_BITS='12',NEUTRINO_ARENA_CODEC_MIN_KIB='16',NEUTRINO_ARENA_OMIT_ANCHOR='1',NEUTRINO_ARENA_HOTPATH='3')
spec={'N':dict(mode=None,env={})}
for key,mode in [('L','legacy_reference'),('C','corrected'),('S','shift'),('A','late'),('V','vector'),('AV','late_vector')]:
    spec[key]=dict(mode='files',env={**base,'NEUTRINO_CORE_MODE':mode})
for key in ['R','T','G']:
    spec[key]=dict(mode='files',env={**base,'NEUTRINO_CORE_MODE':'policy','NEUTRINO_CORE_POLICY':str(root/'policies'/f'{key}.json')})
for c in a.conditions:
    spec[c]['env']['NEUTRINO_BENCH_THREADS']='8'
    if a.diagnostic:spec[c]['env'].update(NEUTRINO_ARENA_DIAG='1',DEEP_SPACE='1')
    if a.validate and c!='N':spec[c]['env'].update(NEUTRINO_ARENA_VERIFY='1',NEUTRINO_CAPTURE_CLEAR='2',NEUTRINO_ARENA_DIAG='1')
    if a.gpu13 and c!='N':
        spec[c]['env'].update(NEUTRINO_ARENA_DELTA='0',NEUTRINO_ARENA_HIER='0',NEUTRINO_ARENA_REFERENCE='0',NEUTRINO_ARENA_OMIT_ANCHOR='0')
        for field in ['NEUTRINO_ARENA_DELTA_WIDTH','NEUTRINO_ARENA_ELAPSED_WIDTH','NEUTRINO_ARENA_LOCAL_BITS']:spec[c]['env'].pop(field)
    if a.gpu:spec[c]['env'].update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',DEEP_CUPTI_CALLBACKS='0',NEUTRINO_PROFILE_COMPAT='1',DEEP_CUPTI_LIBRARY=str(root/'benchmarks/deep_cupti_gpuonly.so'))
sp=out/(a.name+'-specs.json');sp.write_text(json.dumps(spec,indent=2)+'\n')
env=dict(os.environ,PATH='/usr/local/cuda-12.4/bin:'+os.environ.get('PATH',''),NEUTRINO_COMPILER_CACHE_ROOT=str(root/'compiler-cache'))
cmd=[sys.executable,str(root/'benchmarks/run_arena_matrix.py'),'--output',str(out/a.name),'--trace-root','/dev/shm/neutrino-policy-20261004','--spec',str(sp),'--conditions',*a.conditions,'--rounds',str(a.rounds),'--timeout','1200','--compiler-python',str(root/'benchmarks/cross_cache_python')]
if a.diagnostic or a.validate:cmd+=['--diagnostic']
cmd+=['--',sys.executable,str(root/'benchmarks/qwen_arena_workload.py'),'--adapter',str(root/'benchmarks/libqwen_dynamic.so'),'--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin','--model',a.model,'--warmup','2','--requests',str(a.requests),'--prompt-repeat',str(a.prompt_repeat),'--output-tokens',str(a.output_tokens)]
subprocess.run(cmd,cwd=root,env=env,check=True)
