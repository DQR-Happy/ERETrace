"""Isolated, paired tests of the research-roadmap candidates. No raw trace archive."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--conditions',nargs='+',default=['N','O','I','P','Z']);p.add_argument('--rounds',type=int,default=3);p.add_argument('--requests',type=int,default=3);p.add_argument('--repeat',type=int,default=1);p.add_argument('--tokens',type=int,default=32);p.add_argument('--diagnostic',action='store_true');p.add_argument('--validate',action='store_true');p.add_argument('--gpu',action='store_true');p.add_argument('--old-adapter',action='store_true');p.add_argument('--layout',action='store_true');p.add_argument('--model',default='/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf');p.add_argument('--large-pool',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True);py=sys.executable
old=Path('/root/autodl-tmp/neutrino-arena-follow-20261003');ring=Path('/root/autodl-tmp/neutrino-lg64-stall-20261003')
base={'NEUTRINO_ARENA':'1','NEUTRINO_CAPTURE_CLEAR':'1','NEUTRINO_CAPTURE_ENCODER':str(root/'benchmarks/capture_pack.cubin'),'NEUTRINO_ARENA_TOKEN_FLUSH':'1','NEUTRINO_ARENA_HOST_MIB':'32','NEUTRINO_ARENA_DEVICE_MIB':'16','NEUTRINO_ARENA_INCREMENTAL':'1'}
spec={'N':dict(mode=None,env={}), 'O':dict(mode='files',root=str(old),env={**base,'NEUTRINO_CAPTURE_ENCODER':str(old/'benchmarks/capture_pack.cubin')}),'I':dict(mode='files',env=base.copy()),'P':dict(mode='files',env={**base,'NEUTRINO_ARENA_INCREMENTAL':'0','NEUTRINO_ARENA_TEMPLATE':'1'}),'Z':dict(mode='files',root=str(ring),env={'NEUTRINO_ABLATION_ASYNC':'1','NEUTRINO_ABLATION_ASYNC_SLOTS':'64','NEUTRINO_ABLATION_ASYNC_COPY_STREAM':'1','NEUTRINO_CAPTURE_OUTPUT':'3','NEUTRINO_CAPTURE_CLEAR':'1','NEUTRINO_CAPTURE_ENCODER':str(ring/'benchmarks/capture_pack.cubin')})}
def add(k,**e):spec[k]=dict(mode='files',env={**base,**e})
add('W',NEUTRINO_ARENA_WRITER='1')
add('B',NEUTRINO_ARENA_BATCH_ENCODER='1')
add('C',NEUTRINO_ARENA_BATCH_ENCODER='2')
add('BW',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_WRITER='1')
add('BC',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0')
add('BCW',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',NEUTRINO_ARENA_WRITER='1')
add('D',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',NEUTRINO_ARENA_DELTA='1')
add('DW',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',NEUTRINO_ARENA_DELTA='1',NEUTRINO_ARENA_WRITER='1')
add('DF',NEUTRINO_ARENA_BATCH_ENCODER='1',NEUTRINO_ARENA_DECOUPLE='0',NEUTRINO_ARENA_DELTA='1',NEUTRINO_ARENA_FORCE_FALLBACK='1')
for kib in [64,256,4096]:add('S'+str(kib),NEUTRINO_ARENA_STDIO_KIB=str(kib))
for n in [16,256]:add('F'+str(n),NEUTRINO_ARENA_FRAMES=str(n))
for kib in [256,1024,4096]:add('Q'+str(kib),NEUTRINO_ARENA_SOFT_KIB=str(kib))
for age in [50,200,1000]:add('A'+str(age),NEUTRINO_ARENA_SOFT_KIB='1024',NEUTRINO_ARENA_AGE_US=str(age))
spec['ID']=dict(mode='files',diagnostic=True,env={**base,'NEUTRINO_ARENA_DIAG':'1','DEEP_SPACE':'1'})
spec['DWD']=dict(mode='files',diagnostic=True,env={**spec['DW']['env'],'NEUTRINO_ARENA_DIAG':'1','DEEP_SPACE':'1'})
for c in a.conditions:
    if a.large_pool and c not in ['N','O','Z']:spec[c]['env'].update(NEUTRINO_ARENA_HOST_MIB='128',NEUTRINO_ARENA_DEVICE_MIB='64',NEUTRINO_ARENA_CHUNK_MIB='64')
    if a.diagnostic:spec[c]['env'].update(NEUTRINO_ARENA_DIAG='1',DEEP_SPACE='1')
    if a.validate and c not in ['N','O','Z']:spec[c]['env'].update(NEUTRINO_ARENA_VERIFY='1',NEUTRINO_CAPTURE_CLEAR='2',NEUTRINO_ARENA_DIAG='1')
    if a.gpu:spec[c]['env'].update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',NEUTRINO_PROFILE_COMPAT='1',DEEP_CUPTI_CALLBACKS='0',DEEP_CUPTI_LIBRARY='/root/autodl-tmp/neutrino-arena-20261003/benchmarks/deep_cupti_gpuonly.so')
sp=out/(a.name+'-specs.json');sp.write_text(json.dumps(spec,indent=2)+'\n')
cmd=[py,str(root/'benchmarks/run_arena_matrix.py'),'--output',str(out/a.name),'--trace-root','/dev/shm/neutrino-verify-20261004','--spec',str(sp),'--conditions',*a.conditions,'--rounds',str(a.rounds),'--compiler-python',str(root/'benchmarks/cache_python.py')]
if a.diagnostic or a.validate:cmd+=['--diagnostic']
adapter='/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so' if a.old_adapter else str(root/'benchmarks/libqwen_dynamic.so')
work=[py,str(root/'benchmarks/qwen_arena_workload.py'),'--adapter',adapter,'--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin','--model',a.model,'--warmup','2','--requests',str(a.requests),'--output-tokens',str(a.tokens),'--prompt-repeat',str(a.repeat)]
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
if a.validate:env['BULK_RECORD_VALIDITY']='1'
if a.layout:env['ARENA_SAVE_LAYOUT']='1'
subprocess.run([*cmd,'--',*work],cwd=root,env=env,check=True)
