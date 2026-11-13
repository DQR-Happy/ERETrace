"""Resume completed primary suite after the adapter's fixed token buffer limit."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results'
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
failure=dict(stage='context-128-5round',condition='r1-Z',returncode=1,
    error='qwen_prefill returned -100 before model prefill: adapter llama_tokenize destination has fixed capacity 512 tokens',
    replacement='context-64-5round; same adapter and hook binaries; actual token count retained in workload.json')
(out/'context-limit-failure.json').write_text(json.dumps(failure,indent=2)+'\n')
for stage in ['context64','gpu','regression']:
    print('RESUME',stage,flush=True)
    subprocess.run([sys.executable,'benchmarks/run_arena_experiment.py','--stage',stage],cwd=root,env=env,check=True)
(out/'suite-complete.json').write_text(json.dumps(dict(complete=True,excluded_failed_matrix='context-128-5round'))+'\n')
follow=Path('/root/autodl-tmp/neutrino-arena-follow-20261003')
with (follow/'suite.log').open('w') as log:
    subprocess.run([sys.executable,'benchmarks/run_arena_followup.py'],cwd=follow,env=env,stdout=log,stderr=log,check=True)
subprocess.run([sys.executable,'benchmarks/collect_arena_provenance.py'],cwd=follow,env=env,check=True)
