"""Predetermined held-out workload, GPU and correctness follow-ups."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];env=dict(os.environ,PATH='/usr/local/cuda-12.4/bin:'+os.environ.get('PATH',''),PYTHONPATH=str(root))
def run(args):
    print('RUN',args,flush=True);subprocess.run([sys.executable,*args],cwd=root,env=env,check=True)
assert (root/'results/policy-heldout/complete.json').exists()
run(['benchmarks/run_core_policy_matrix.py','--name','policy-valid','--conditions','G','R','T','--rounds','1','--requests','1','--validate'])
run(['benchmarks/run_core_policy_matrix.py','--name','policy-gpu-test','--conditions','S','A','R','T','--rounds','5','--requests','1','--gpu13','--gpu'])
run(['benchmarks/run_core_policy_matrix.py','--name','policy-long','--conditions','N','S','R','T','--rounds','5','--requests','3','--prompt-repeat','95'])
run(['benchmarks/run_core_policy_matrix.py','--name','policy-model17','--conditions','N','S','R','T','--rounds','5','--requests','3','--model','/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-1.7B-Q4_K_M.gguf'])
(root/'results/policy-followup-complete.json').write_text('{"passed":true}\n')
