"""Run final adapter/codec GPU checks after all performance matrices finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--nvcomp',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--arch',default='sm_86')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1];cuda=Path('/usr/local/cuda')
exe=a.output/'layout-test'
cmd=[str(cuda/'bin/nvcc'),'-O3','-std=c++17','-arch='+a.arch,
     str(root/'benchmarks/nvpipe_layout_test.cu'),'-I'+str(a.nvcomp/'include'),
     '-L'+str(a.nvcomp/'lib64'),'-Xlinker','-rpath','-Xlinker',str(a.nvcomp/'lib64'),
     '-l:libnvcomp.so.5','-o',str(exe)]
subprocess.run(cmd,check=True,cwd=root)
results={}
for tool in ['memcheck','racecheck','synccheck']:
    with (a.output/(tool+'.json')).open('w') as f:
        subprocess.run([str(cuda/'compute-sanitizer/compute-sanitizer'),'--tool',tool,
                        '--error-exitcode','99','--log-file',str(a.output/(tool+'.log')),str(exe)],
                       stdout=f,check=True,cwd=root)
    results[tool]=json.loads((a.output/(tool+'.json')).read_text())
    assert results[tool]['passed']
subprocess.run([sys.executable,str(root/'benchmarks/gpu_compact_smoke.py'),
                str(root/'benchmarks/capture_pack.cubin'),str(a.output/'compact.json')],check=True,cwd=root)
results['compact']=json.loads((a.output/'compact.json').read_text())
assert results['compact']['passed']
(a.output/'checks.json').write_text(json.dumps(dict(passed=True,command=cmd,results=results),indent=2)+'\n')
