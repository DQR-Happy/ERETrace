"""Final diagnostic controls, CPU checks, and immutable evidence collection."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];py=sys.executable
v3=Path('/root/autodl-tmp/neutrino-hotpath-20261004')
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
deadline=time.monotonic()+3600
while not (root/'results/final-suite-complete.json').exists():
 if time.monotonic()>deadline:raise RuntimeError('Main suite did not complete')
 time.sleep(10)
def cmd(r,name,args):
 print(name,'START',flush=True)
 with (r/(name+'.log')).open('w') as f:subprocess.run(args,cwd=r,env=env,stdout=f,stderr=f,check=True)
 print(name,'DONE',flush=True)
cmd(root,'final-gpu',[py,'benchmarks/run_codec_comparison_20261004.py','--name','final-gpu','--conditions','N','K16','HOT','NSB','--rounds','3','--requests','1','--gpu'])
cmd(root,'final-observer',[py,'benchmarks/run_codec_comparison_20261004.py','--name','final-observer','--conditions','HOT','HOTD','NSB','NSBD','--rounds','5'])
cmd(v3,'cpu-regression',[py,'-m','unittest','discover','-s','tests','-p','test_*.py'])
for name in ['neutrino-codec-comparison-20261004','neutrino-nvcomp-pipeline-20261004','neutrino-hotpath-20261004','neutrino-nvcomp-tuned-20261004']:
 r=Path('/root/autodl-tmp')/name
 cmd(r,'provenance',[py,'benchmarks/verification_provenance.py'])
 remaining=list(Path('/dev/shm/neutrino-codec-comparison-20261004').rglob('*.nts'))+list(Path('/dev/shm/neutrino-codec-comparison-20261004').rglob('*.bin'))
 (r/'results/cleanup.json').write_text(json.dumps(dict(raw_trace_remaining=len(remaining),paths=[str(p) for p in remaining]),indent=2)+'\n')
 assert not remaining
(root/'results/evidence-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
