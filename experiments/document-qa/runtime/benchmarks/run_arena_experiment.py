"""Finite serial matrices; each process writes complete trace then verifies/deletes it."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','final-smoke','main','detail','long','context','context64','observer','gpu','regression'],required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True)
old=Path('/root/autodl-tmp/neutrino-lg64-stall-20261003')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
ring={'NEUTRINO_ABLATION_ASYNC':'1','NEUTRINO_ABLATION_ASYNC_SLOTS':'64','NEUTRINO_ABLATION_ASYNC_COPY_STREAM':'1','NEUTRINO_CAPTURE_OUTPUT':'3','NEUTRINO_CAPTURE_CLEAR':'1','NEUTRINO_CAPTURE_ENCODER':str(root/'benchmarks/capture_pack.cubin')}
base={'NEUTRINO_ARENA':'1','NEUTRINO_CAPTURE_CLEAR':'1','NEUTRINO_CAPTURE_ENCODER':str(root/'benchmarks/capture_pack.cubin')}
spec={'N':dict(mode=None,env={}), 'Z':dict(mode='files',root=str(old),env={**ring,'NEUTRINO_CAPTURE_ENCODER':str(old/'benchmarks/capture_pack.cubin')}),'Znew':dict(mode='files',env=ring)}
for name,extra in {
 'C1':{'NEUTRINO_ARENA_FRAMES':'1','NEUTRINO_ARENA_DECOUPLE':'0'},
 'D1':{'NEUTRINO_ARENA_FRAMES':'1'},
 'C64':{'NEUTRINO_ARENA_DECOUPLE':'0'},'D64':{},
 'F64':{'NEUTRINO_ARENA_FRAME_COPIES':'1'},
 'T64':{'NEUTRINO_ARENA_TOKEN_FLUSH':'1'},
 'D256':{'NEUTRINO_ARENA_FRAMES':'256'},
 'P64':{'NEUTRINO_ARENA_TEMPLATE':'1'},
 'P16m':{'NEUTRINO_ARENA_TEMPLATE':'1','NEUTRINO_ARENA_HOST_MIB':'32','NEUTRINO_ARENA_DEVICE_MIB':'16'},
 'VP':{'NEUTRINO_ARENA_TEMPLATE':'1','NEUTRINO_ARENA_VERIFY':'1','NEUTRINO_ARENA_DIAG':'1','NEUTRINO_CAPTURE_CLEAR':'2'},
 'D16m':{'NEUTRINO_ARENA_HOST_MIB':'32','NEUTRINO_ARENA_DEVICE_MIB':'16'},
 'D32m':{'NEUTRINO_ARENA_HOST_MIB':'64','NEUTRINO_ARENA_DEVICE_MIB':'32'},
 'V64':{'NEUTRINO_ARENA_VERIFY':'1','NEUTRINO_ARENA_DIAG':'1','NEUTRINO_CAPTURE_CLEAR':'2'},
}.items():spec[name]=dict(mode='files',env={**base,**extra})
def matrix(name,conditions,rounds=7,requests=3,repeat=1,diag=False,records=False):
    dest=out/name
    sp=out/(name+'-specs.json');sp.write_text(json.dumps(spec,indent=2)+'\n')
    cmd=[sys.executable,'benchmarks/run_arena_matrix.py','--output',str(dest),'--trace-root','/dev/shm/neutrino-arena-20261003','--spec',str(sp),'--conditions',*conditions,'--rounds',str(rounds),'--compiler-python',str(root/'benchmarks/cache_python.py')]
    if diag:cmd+=['--diagnostic']
    work=[sys.executable,str(root/'benchmarks/qwen_arena_workload.py'),'--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so','--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin','--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf','--warmup','2','--requests',str(requests),'--output-tokens','32','--prompt-repeat',str(repeat)]
    e=dict(env)
    if records:e['BULK_RECORD_VALIDITY']='1'
    subprocess.run([*cmd,'--',*work],cwd=root,env=e,check=True)
if a.stage=='smoke':matrix('smoke-v2',['N','Znew','C1','D1','C64','D64','F64','T64','D256','D16m','V64','P64','P16m','VP'],1,1,diag=True,records=True)
if a.stage=='final-smoke':matrix('release-smoke',['N','Znew','D64','P16m','VP'],1,1,diag=True,records=True)
if a.stage=='main':matrix('main-7round',['N','Z','Znew','C1','D1','C64','D64','F64','T64','D256','D16m','D32m','P64','P16m'])
if a.stage=='detail':
    for c in ['C1','D1','C64','D64','F64','T64','D256','D16m','D32m','P64','P16m']:spec[c]['env']['NEUTRINO_ARENA_DIAG']='1'
    matrix('detail-3round',['C1','D1','C64','D64','F64','T64','D256','D16m','D32m','P64','P16m'],3,1,diag=True)
if a.stage=='long':matrix('long-3round',['N','Z','D64','D256','P16m'],3,20)
if a.stage=='context':
    for repeat in [16,64]:matrix(f'context-{repeat}-5round',['N','Z','D64','D256','P16m'],5,3,repeat)
if a.stage=='context64':matrix('context-64-5round',['N','Z','D64','D256','P16m'],5,3,64)
if a.stage=='regression':subprocess.run([sys.executable,'tests/run_gpu_validation.py','--output',str(out/'default-regression'),'--trace-root','/dev/shm/neutrino-arena-20261003'],cwd=root,env=env,check=True)

if a.stage=='observer':
    spec['DD']=dict(mode='files',env={**base,'NEUTRINO_ARENA_DIAG':'1'})
    spec['PD']=dict(mode='files',env={**spec['P16m']['env'],'NEUTRINO_ARENA_DIAG':'1'})
    matrix('observer-5round',['D64','DD','P16m','PD'],5,3)
if a.stage=='gpu':
    for c in ['N','Znew','D1','D64','F64','P16m']:
        spec[c]['env'].update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',NEUTRINO_PROFILE_COMPAT='1',DEEP_CUPTI_CALLBACKS='0',DEEP_CUPTI_LIBRARY=str(root/'benchmarks/deep_cupti_gpuonly.so'))
    matrix('gpu-3round',['N','Znew','D1','D64','F64','P16m'],3,1)
