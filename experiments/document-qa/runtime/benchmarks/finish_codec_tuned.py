"""Run isolated confirmation and validation sequentially after the V3 matrix."""
import json,os,statistics,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];py=sys.executable
v3=Path('/root/autodl-tmp/neutrino-hotpath-20261004')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
deadline=time.monotonic()+2400
while not (v3/'results/confirmation-complete.json').exists():
 if time.monotonic()>deadline:raise RuntimeError('Previous suite did not complete; no GPU overlap allowed')
 time.sleep(10)
(root/'results').mkdir(exist_ok=True)
def command(r,name,args):
 print(name,'START',flush=True)
 with (r/(name+'.log')).open('w') as f:subprocess.run(args,cwd=r,env=env,stdout=f,stderr=f,check=True)
 print(name,'DONE',flush=True)
def matrix(name,conds,rounds,*extra):
 command(root,name,[py,'benchmarks/run_codec_comparison_20261004.py','--name',name,'--conditions',*conds.split(),'--rounds',str(rounds),*extra])
command(root,'nvpipe-tuned-smoke',[py,'benchmarks/gpu_nvpipe_smoke.py','benchmarks/nvcomp_pipeline.so','results/nvpipe-tuned-smoke.json'])
command(root,'nvpipe-tuned-memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'benchmarks/gpu_nvpipe_smoke.py','benchmarks/nvcomp_pipeline.so','results/nvpipe-tuned-memcheck.json'])
matrix('tuned-validation','NS NS64T NSB',1,'--requests','1','--tokens','4','--validate')
matrix('tuned-screen','N K16 NSOLD NS NS64T NSB KB HOT',3)
runs=json.loads((root/'results/tuned-screen/runs.json').read_text())
means={c:statistics.mean(r['wall_ns']/len(r['requests']) for r in runs if r['condition']==c) for c in ['NS','NS64T','NSB']}
best=min(means,key=means.get)
(root/'results/nvcomp-selection.json').write_text(json.dumps(dict(rule='lowest process mean among NS/NS64T/NSB in the independent screen',means_ns=means,selected=best),indent=2)+'\n')
matrix('final-confirmation','N KOLD K16 HOT '+best,7)
matrix('tuned-host','K16 HOT NSOLD NS NS64T NSB',3,'--requests','1','--diagnostic')
command(v3,'build-hotshape',['nvcc','-O2','-arch=sm_86','benchmarks/hotpath_shape_workload.cu','-ldl','-o','benchmarks/hotpath_shape_workload'])
command(v3,'hotshape-check',[py,'benchmarks/run_hotpath_shapes.py'])
command(v3,'stateful-check',[py,'benchmarks/run_codec_stateful.py'])
command(v3,'compact-memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','99',py,'benchmarks/gpu_compact_smoke.py','benchmarks/capture_pack.cubin','results/compact-memcheck.json'])
command(v3,'default-gpu-regression',[py,'tests/run_gpu_validation.py','--output','results/default-gpu-regression','--trace-root','/dev/shm/neutrino-codec-comparison-20261004'])
matrix('final-model17','N KOLD HOT '+best,5,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-1.7B-Q4_K_M.gguf')
matrix('final-llama1b','N KOLD HOT',5,'--model','/autodl-fs/data/qwen3-small-aiperf/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf')
matrix('final-context800','N KOLD HOT',5,'--repeat','128')
matrix('final-sustained','N KOLD HOT',3,'--requests','20')
(root/'results/final-suite-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
