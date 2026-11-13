"""Strong generic controls for the learned layout, in an isolated second build."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['build','validation','main','detail','confirmation','long','context','gpu','all'],default='all');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True);py=sys.executable
primary=Path('/root/autodl-tmp/neutrino-arena-20261003');old=Path('/root/autodl-tmp/neutrino-lg64-stall-20261003')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
base={'NEUTRINO_ARENA':'1','NEUTRINO_CAPTURE_CLEAR':'1','NEUTRINO_CAPTURE_ENCODER':str(root/'benchmarks/capture_pack.cubin')}
small={**base,'NEUTRINO_ARENA_TOKEN_FLUSH':'1','NEUTRINO_ARENA_HOST_MIB':'32','NEUTRINO_ARENA_DEVICE_MIB':'16'}
ring={'NEUTRINO_ABLATION_ASYNC':'1','NEUTRINO_ABLATION_ASYNC_SLOTS':'64','NEUTRINO_ABLATION_ASYNC_COPY_STREAM':'1','NEUTRINO_CAPTURE_OUTPUT':'3','NEUTRINO_CAPTURE_CLEAR':'1','NEUTRINO_CAPTURE_ENCODER':str(old/'benchmarks/capture_pack.cubin')}
spec={'N':dict(mode=None,env={}), 'Z':dict(mode='files',root=str(old),env=ring),
 'Aold':dict(mode='files',root=str(primary),env={**base,'NEUTRINO_CAPTURE_ENCODER':str(primary/'benchmarks/capture_pack.cubin')}),
 'A':dict(mode='files',env=base),'G16':dict(mode='files',env=small),
 'S16':dict(mode='files',env={**small,'NEUTRINO_ARENA_TRIM':'1'}),
 'I16':dict(mode='files',env={**small,'NEUTRINO_ARENA_INCREMENTAL':'1'}),
 'P16':dict(mode='files',env={**small,'NEUTRINO_ARENA_TEMPLATE':'1'}),
 'I128':dict(mode='files',env={**base,'NEUTRINO_ARENA_INCREMENTAL':'1','NEUTRINO_ARENA_TOKEN_FLUSH':'1'}),
 'I10':dict(mode='files',env={**small,'NEUTRINO_ARENA_INCREMENTAL':'1','NEUTRINO_ARENA_CHUNK_MIB':'10','NEUTRINO_ARENA_HOST_MIB':'20','NEUTRINO_ARENA_DEVICE_MIB':'10'})}
def matrix(name,conditions,rounds=7,requests=3,repeat=1,diag=False,records=False):
    sp=out/(name+'-specs.json');sp.write_text(json.dumps(spec,indent=2)+'\n')
    cmd=[py,'benchmarks/run_arena_matrix.py','--output',str(out/name),'--trace-root','/dev/shm/neutrino-arena-follow','--spec',str(sp),'--conditions',*conditions,'--rounds',str(rounds),'--compiler-python',str(root/'benchmarks/cache_python.py')]
    if diag:cmd+=['--diagnostic']
    work=[py,str(root/'benchmarks/qwen_arena_workload.py'),'--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so','--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin','--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf','--warmup','2','--requests',str(requests),'--output-tokens','32','--prompt-repeat',str(repeat)]
    e=dict(env)
    if records:e['BULK_RECORD_VALIDITY']='1'
    subprocess.run([*cmd,'--',*work],cwd=root,env=e,check=True)
if a.stage in ('all','build'):
    subprocess.run([py,'neutrino/build.py'],cwd=root,env=env,check=True)
    subprocess.run(['cc','-shared','-fPIC','-O3','benchmarks/profile_preload.c','-ldl','-o','neutrino/build/profile-preload.so'],cwd=root,env=env,check=True)
    subprocess.run([py,'-m','unittest','discover','-s','tests','-p','test_arena_ring.py'],cwd=root,env=env,check=True)
if a.stage in ('all','validation'):
    subprocess.run([py,'benchmarks/run_arena_stateful.py'],cwd=root,env=env,check=True)
    for c in ['S16','I16','I10']:spec[c]['env'].update(NEUTRINO_CAPTURE_CLEAR='2',NEUTRINO_ARENA_VERIFY='1',NEUTRINO_ARENA_DIAG='1')
    matrix('generic-validation',['S16','I16','I10'],1,1,diag=True,records=True)
    for c in ['S16','I16','I10']:
        spec[c]['env']['NEUTRINO_CAPTURE_CLEAR']='1';spec[c]['env'].pop('NEUTRINO_ARENA_VERIFY');spec[c]['env'].pop('NEUTRINO_ARENA_DIAG')
if a.stage in ('all','main'):matrix('generic-main-7round',list(spec))
if a.stage in ('all','detail'):
    for c in ['A','G16','S16','I16','P16','I128','I10']:spec[c]['env'].update(NEUTRINO_ARENA_DIAG='1',DEEP_SPACE='1')
    matrix('generic-detail-3round',['A','G16','S16','I16','P16','I128','I10'],3,1,diag=True)
    for c in ['A','G16','S16','I16','P16','I128','I10']:spec[c]['env'].pop('NEUTRINO_ARENA_DIAG');spec[c]['env'].pop('DEEP_SPACE')
if a.stage in ('all','confirmation'):matrix('confirmation-7round',['N','Z','Aold','I16','P16'])
if a.stage in ('all','long'):matrix('generic-long-3round',['N','Z','I16','P16'],3,20)
if a.stage in ('all','context'):
    for repeat in [16,64]:matrix(f'generic-context-{repeat}-5round',['N','Z','I16','P16'],5,3,repeat)
if a.stage in ('all','gpu'):
    for c in ['N','Z','I16','P16']:
        spec[c]['env'].update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',NEUTRINO_PROFILE_COMPAT='1',DEEP_CUPTI_CALLBACKS='0',DEEP_CUPTI_LIBRARY=str(primary/'benchmarks/deep_cupti_gpuonly.so'))
    matrix('generic-gpu-3round',['N','Z','I16','P16'],3,1)
if a.stage=='all':
    subprocess.run([py,'tests/run_gpu_validation.py','--output',str(out/'generic-default-regression'),'--trace-root','/dev/shm/neutrino-arena-follow'],cwd=root,env=env,check=True)
    (out/'suite-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
