"""Finite serial diagnostics after the independent main performance matrix."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];env=dict(os.environ,PATH='/usr/local/cuda-12.4/bin:'+os.environ.get('PATH',''),PYTHONPATH=str(root));py=sys.executable
assert (root/'results/core-confirmation/complete.json').exists()
def run(args):
    print('RUN',args,flush=True);subprocess.run(list(map(str,args)),cwd=root,env=env,check=True)
run([py,'benchmarks/core_event_validation.py','--output','results/events'])
run(['compute-sanitizer','--tool','memcheck','--target-processes','all','--error-exitcode','86','--log-file',str(root/'results/event-memcheck.txt'),py,'benchmarks/core_event_validation.py','--output','results/events-memcheck'])
for r in range(1,8):
    run([py,'benchmarks/core_gpu_experiment.py','--output','results/core-micro-confirmation','--reference','reference/cuda.py','--round',str(r)]+(['--build'] if r==1 else []))
cuda='/usr/local/cuda/targets/x86_64-linux'
run(['cc','-shared','-fPIC','-O3','benchmarks/profile_preload.c','-ldl','-o','neutrino/build/profile-preload.so'])
run(['cc','-shared','-fPIC','-O3','benchmarks/deep_cupti_gpuonly.c','-I'+cuda+'/include','-L'+cuda+'/lib','-Wl,-rpath,'+cuda+'/lib','-lcupti','-ldl','-lpthread','-o','benchmarks/deep_cupti_gpuonly.so'])
run([py,'benchmarks/run_core_matrix.py','--name','core-host','--conditions','L','C','S','A','AV','--rounds','3','--requests','1','--diagnostic'])
run([py,'benchmarks/run_core_matrix.py','--name','core-gpu','--rounds','3','--requests','1','--gpu'])
run([py,'benchmarks/collect_core_evidence.py','--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf'])
(root/'results/followup-complete.json').write_text('{"passed":true}\n')
