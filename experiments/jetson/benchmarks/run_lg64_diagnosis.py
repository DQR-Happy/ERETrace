"""Finite isolated LG64 residual-overhead experiments, serial on one GPU."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--resume',action='store_true')
ap.add_argument('--step',choices=['build','validation','main','perturbation','detail','cupti','regression','all'],default='all')
args=ap.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True);py=sys.executable
old=Path('/root/autodl-tmp/neutrino-ring-output-20261003')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
    NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
base={'NEUTRINO_ABLATION_ASYNC':'1','NEUTRINO_ABLATION_ASYNC_SLOTS':'64',
      'NEUTRINO_ABLATION_ASYNC_COPY_STREAM':'1','NEUTRINO_CAPTURE_OUTPUT':'3',
      'NEUTRINO_CAPTURE_ENCODER':str(root/'benchmarks/capture_pack.cubin')}
spec={'N':dict(mode=None,env={}), 'H':dict(mode='files',root=str(old),env={**base,
    'NEUTRINO_CAPTURE_ENCODER':str(old/'benchmarks/capture_pack.cubin')})}
for c,extra in {'G':{},'R':{'NEUTRINO_CAPTURE_CONSUME':'1'},'X':{'NEUTRINO_CAPTURE_CONSUME':'2'},
        'Y':{'NEUTRINO_CAPTURE_CONSUME':'2','NEUTRINO_CAPTURE_NO_D2H':'1'},
        'O':{'NEUTRINO_CAPTURE_KERNEL':'0'},'P':{'NEUTRINO_CAPTURE_KERNEL':'1'},
        'Z':{'NEUTRINO_CAPTURE_CLEAR':'1'},'V':{'NEUTRINO_CAPTURE_VERIFY_ENCODER':'1'},
        'D':{'NEUTRINO_CAPTURE_DETAIL':'1'},'DC':{'NEUTRINO_CAPTURE_DETAIL':'1','NEUTRINO_CAPTURE_DETAIL_CPU':'1'}}.items():
    spec[c]=dict(mode='files',env={**base,**extra})
floor={**base,'NEUTRINO_CAPTURE_CONSUME':'2','NEUTRINO_CAPTURE_NO_D2H':'1','NEUTRINO_CAPTURE_NO_PACK':'1'}
spec['U']=dict(mode='files',env=floor)
spec['A']=dict(mode='files',env={**floor,'NEUTRINO_CAPTURE_KERNEL':'0'})
spec['AP']=dict(mode='files',env={**floor,'NEUTRINO_CAPTURE_KERNEL':'1'})
spec['AZ']=dict(mode='files',env={**spec['A']['env'],'NEUTRINO_CAPTURE_CLEAR':'1'})
sp=out/'specs.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
work=[py,str(root/'benchmarks/qwen_capture_workload.py'),'--adapter','/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
    '--backend','/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
    '--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
    '--requests','3','--warmup','2','--output-tokens','32']
def matrix(name,conditions,rounds=7,diag=False,requests=3,records=False,specpath=sp):
    dest=out/name
    if args.resume and (dest/'complete.json').exists():
        c=json.loads((dest/'config.json').read_text())
        assert c['conditions']==conditions and c['rounds']==rounds and c['diagnostic']==diag
        print('VERIFIED COMPLETE',name,flush=True);return
    w=list(work);w[w.index('--requests')+1]=str(requests)
    cmd=[py,'benchmarks/run_bulk_matrix.py','--output',str(dest),'--trace-root','/dev/shm/neutrino-lg64-detail',
         '--spec',str(specpath),'--conditions',*conditions,'--rounds',str(rounds),
         '--compiler-python',str(root/'benchmarks/cache_python.py')]
    if diag:cmd+=['--diagnostic']
    e=dict(env)
    if records:e['BULK_RECORD_VALIDITY']='1'
    print('LG64 START',name,flush=True);subprocess.run([*cmd,'--',*w],cwd=root,env=e,check=True)
    print('LG64 DONE',name,flush=True)
def build():
    subprocess.run([py,'neutrino/build.py'],cwd=root,env=env,check=True)
    subprocess.run(['cc','-shared','-fPIC','-O3','benchmarks/profile_preload.c','-ldl','-o','neutrino/build/profile-preload.so'],cwd=root,env=env,check=True)
    subprocess.run(['nvcc','-O3','-arch=sm_86','--cubin','benchmarks/capture_pack.cu','-o','benchmarks/capture_pack.cubin'],cwd=root,env=env,check=True)
    cuda='/usr/local/cuda/targets/x86_64-linux'
    subprocess.run(['cc','-shared','-fPIC','-O3','benchmarks/deep_cupti.c','-I'+cuda+'/include','-L'+cuda+'/lib',
        '-Wl,-rpath,'+cuda+'/lib','-lcupti','-ldl','-lpthread','-o','benchmarks/deep_cupti.so'],cwd=root,env=env,check=True)
    subprocess.run([py,'-m','unittest','discover','-s','tests','-p','test_capture_detail.py'],cwd=root,env=env,check=True)
if args.step in ('all','build'):build()
if args.step in ('all','validation'):
    matrix('validation',['G','R','Z','V'],1,True,1,True)
    matrix('controls-smoke',['X','Y','O','P'],1,True,1)
    matrix('floor-smoke',['U','A','AP','AZ'],1,True,1)
    matrix('validation-final',['G','R','Z','V'],1,True,1,True)
if args.step in ('all','main'):matrix('main-7round',['N','H','G','R','X','Y','U','A','AP','AZ','Z'])
if args.step in ('all','perturbation'):matrix('observer-5round',['G','D','DC'],5)
if args.step in ('all','detail'):
    detail=copy.deepcopy(spec)
    for c in ['G','R','X','Y','U','A','AP','AZ','Z']:
        detail[c]['env'].update(NEUTRINO_CAPTURE_DETAIL='1',NEUTRINO_CAPTURE_DETAIL_CPU='1',DEEP_SPACE='1')
    detail['N']['env']['DEEP_SPACE']='1'
    dp=out/'detail-specs.json';dp.write_text(json.dumps(detail,indent=2)+'\n')
    matrix('detail-3round',['N','G','R','X','Y','U','A','AP','AZ','Z'],3,True,1,specpath=dp)
if args.step in ('all','cupti'):
    profile=copy.deepcopy(spec)
    for c in ['N','G','X','Y','U','A','AP','AZ','Z']:
        profile[c]['env'].update(DEEP_CUPTI='1',DEEP_CUPTI_ACTIVITY='1',NEUTRINO_PROFILE_COMPAT='1')
    pp=out/'profile-specs.json';pp.write_text(json.dumps(profile,indent=2)+'\n')
    matrix('cupti-3round',['N','G','X','Y','U','A','AP','AZ','Z'],3,False,1,specpath=pp)
if args.step in ('all','regression'):
    subprocess.run([py,'tests/run_gpu_validation.py','--output',str(out/'default-regression'),
        '--trace-root','/dev/shm/neutrino-lg64-detail'],cwd=root,env=env,check=True)
if args.step=='all':(out/'complete.json').write_text(json.dumps(dict(complete=True))+'\n')
