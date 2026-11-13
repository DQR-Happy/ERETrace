"""Run microbenchmarks only after inference matrices have released the GPU/host."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results';py=sys.executable;env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''))
start=time.monotonic();absent=0
while not (out/'model17-5round/complete.json').exists():
    running=subprocess.run(['pgrep','-f','[r]un_verification_20261004.py'],stdout=subprocess.DEVNULL).returncode==0
    absent=0 if running else absent+1
    if absent>6 or time.monotonic()-start>5400:raise RuntimeError('Main matrices stopped or timed out before completion; inspect logs')
    time.sleep(5)
def call(cmd,**kw):subprocess.run(cmd,cwd=root,env=env,check=True,**kw)
call([py,'benchmarks/run_verification_20261004.py','--name','output64-5round','--conditions','N','I','DW','--rounds','5','--tokens','64'])
call([py,'benchmarks/run_verification_20261004.py','--name','old-adapter-control','--conditions','N','--rounds','1','--old-adapter'])
call(['cc','-O3','benchmarks/arena_writer_micro.c','-o','benchmarks/arena_writer_micro'])
temp=Path('/dev/shm/neutrino-writer-verify-20261004');temp.mkdir(exist_ok=False)
try:
    with (out/'writer-micro.json').open('w') as f:call([str(root/'benchmarks/arena_writer_micro'),str(out/'host-detail-3round/r1-I/frame-layout.tsv'),str(temp),'7'],stdout=f)
finally:shutil.rmtree(temp)
call([py,'benchmarks/mmap_register_probe.py',str(out/'mmap-register.json')])
with (out/'regression-tests.txt').open('w') as f:
    for test in ['test_timing.py','test_arena_ring.py','test_capture_codec.py','test_gpu9_format.py']:call([py,'-m','unittest','discover','-s','tests','-p',test],stdout=f,stderr=f)
call([py,'tests/run_gpu_validation.py','--output',str(out/'default-gpu-regression'),'--trace-root','/dev/shm/neutrino-verify-20261004'])
call([py,'benchmarks/verification_provenance.py'])
rawroot=Path('/dev/shm/neutrino-verify-20261004');assert rawroot.exists() and not any(rawroot.iterdir())
(out/'cleanup.json').write_text(json.dumps(dict(raw_trace_remaining=0,root=str(rawroot),mmap_probe_removed=True,writer_micro_removed=True))+'\n')
(out/'suite-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
