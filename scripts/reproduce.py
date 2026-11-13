#!/usr/bin/env python3
"""Resolve portable paper configurations, record inputs, and launch paired runs."""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256')
    p.add_argument('--backend',type=Path,required=True,help='llama.cpp build/bin directory')
    p.add_argument('--adapter',type=Path,default=ROOT/'build/libllama_adapter.so')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trace-root',type=Path,default=Path('/dev/shm/neutrino-artifact'))
    p.add_argument('--conditions',nargs='+',default=['native','ours'])
    p.add_argument('--rounds',type=int,default=7);p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--requests',type=int,default=3);p.add_argument('--input-tokens',type=int,default=256)
    p.add_argument('--output-tokens',type=int,default=64);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--context',type=int,default=4096);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--seed',type=int,default=21042);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.batch_size not in (1,2,4,8,16):p.error('Supported batches: 1/2/4/8/16')
    workload=ROOT/'benchmarks/exact_arena_workload.py'
    if a.batch_size==16:
        workload=ROOT/'experiments/b16/workload.py'
        if a.adapter==ROOT/'build/libllama_adapter.so':a.adapter=ROOT/'build/libllama_adapter_b16.so'
    if min(a.rounds,a.requests,a.input_tokens,a.output_tokens,a.threads)<1 or a.warmup<0:p.error('Positive sizes required')
    if a.batch_size*(a.input_tokens+a.output_tokens)>a.context:p.error('Total sequence token budget exceeds --context')
    specs=json.loads((ROOT/'configs/conditions.json').read_text())
    if not a.conditions or len(set(a.conditions))!=len(a.conditions) or any(x not in specs for x in a.conditions):p.error('Use distinct supported conditions: '+', '.join(specs))
    selected={k:specs[k] for k in a.conditions}
    for s in selected.values():
        s.update(root=str(ROOT),diagnostic=a.diagnostic)
        e=s['env'];e.update(NEUTRINO_BENCH_CONTEXT=str(a.context),NEUTRINO_BENCH_SEQUENCES=str(a.batch_size),NEUTRINO_BENCH_THREADS=str(a.threads))
        if e.get('NEUTRINO_ARENA'):e['NEUTRINO_CAPTURE_ENCODER']=str(ROOT/'build/capture_pack.cubin')
        if e.get('NEUTRINO_ARENA_NVCOMP'):e['NEUTRINO_NVPIPE_LIBRARY']=str(ROOT/'build/nvcomp_pipeline.so')
    command=[sys.executable,str(ROOT/'benchmarks/run_arena_matrix.py'),'--output',str(a.output.resolve()),'--trace-root',str(a.trace_root.resolve()),'--rounds',str(a.rounds),'--seed',str(a.seed),'--timeout',str(a.timeout),'--conditions',*a.conditions,'--spec','SPEC.json','--',sys.executable,str(workload),'--adapter',str(a.adapter.resolve()),'--backend',str(a.backend.resolve()),'--model',str(a.model.resolve()),'--warmup',str(a.warmup),'--requests',str(a.requests),'--input-tokens',str(a.input_tokens),'--output-tokens',str(a.output_tokens),'--batch-size',str(a.batch_size)]
    if a.dry_run:
        print(json.dumps({'specs':selected,'command':command},indent=2));return
    if platform.system()!='Linux':p.error('GPU experiments require Linux; use --dry-run for configuration inspection')
    if a.output.exists():p.error('Output must be a new directory; existing measurements are never overwritten')
    required=[a.model,a.adapter,a.backend/'libllama.so',ROOT/'neutrino/build/config.toml',ROOT/'build/capture_pack.cubin']
    if 'nvcomp' in a.conditions:required.append(ROOT/'build/nvcomp_pipeline.so')
    for f in required:
        if not f.is_file():p.error('Missing '+str(f))
    actual=digest(a.model)
    if a.model_sha256 and a.model_sha256.lower()!=actual:p.error('Model SHA-256 mismatch')
    a.trace_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.trace_root).free<6*1024**3:p.error('At least 6 GiB free trace storage is required for the short primary workload; larger cases need more')
    manifest={'model_sha256':actual,'inputs':{str(f.resolve()):digest(f) for f in required if f!=a.model},'platform':platform.platform(),'python':sys.version,'diagnostic':a.diagnostic,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='neutrino-spec-') as td:
        spec=Path(td)/'spec.json';spec.write_text(json.dumps(selected,indent=2))
        command[command.index('SPEC.json')]=str(spec)
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
        rc=subprocess.run(command,cwd=ROOT,env=env).returncode
    if a.output.exists():(a.output/'environment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if rc:raise SystemExit(rc)
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_runs.py'),str(a.output)],check=True)
if __name__=='__main__':main()
