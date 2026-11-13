"""Finite paper-closure suite. Existing output directories are never replaced."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--models',type=Path,required=True)
p.add_argument('--backend',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=True)
plans=[
 ('confirmation','N G R T RB VS VP',7,{},'Qwen3-0.6B-Q4_K_M.gguf'),
 ('original-anchor','N S R',3,{},'Qwen3-0.6B-Q4_K_M.gguf'),
 ('heldout-llama','N G R VP',5,{},'Llama-3.2-1B-Instruct-Q4_K_M.gguf'),
 ('heldout-qwen17','N R VP',5,{},'Qwen3-1.7B-Q4_K_M.gguf'),
 ('heldout-output64','N R VP',5,{'tokens':64},'Qwen3-0.6B-Q4_K_M.gguf'),
 ('host-detail','G R VS VP',3,{'diagnostic':True,'requests':1},'Qwen3-0.6B-Q4_K_M.gguf'),
 ('gpu-detail','N G R VS VP',3,{'gpu':True,'requests':1},'Qwen3-0.6B-Q4_K_M.gguf'),
 ('query-exact','R RF',1,{'queries':True,'requests':1},'Qwen3-0.6B-Q4_K_M.gguf'),
 ('nvcomp-exact','VP',1,{'validate':True,'requests':1},'Qwen3-0.6B-Q4_K_M.gguf'),
]
(a.output/'plan.json').write_text(json.dumps(plans,indent=2)+'\n')
done=[]
for name,conditions,rounds,options,model in plans:
    cmd=[sys.executable,str(root/'benchmarks/run_final_closure.py'),'--output',str(a.output/name),
         '--model',str(a.models/model),'--backend',str(a.backend),'--conditions',*conditions.split(),
         '--rounds',str(rounds)]
    for k,v in options.items():
        cmd+=['--'+k]
        if v is not True:cmd+=[str(v)]
    print('MATRIX',name,flush=True)
    subprocess.run(cmd,cwd=root,check=True)
    done.append(name);(a.output/'completed-matrices.json').write_text(json.dumps(done)+'\n')
print('FINAL FINITE SUITE COMPLETE',flush=True)
