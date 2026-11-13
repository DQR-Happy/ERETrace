"""Hash actual binaries/data; retain compact device evidence, never model/trace payload."""
import argparse, ctypes as C, hashlib, json, platform, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--backend',type=Path,required=True);p.add_argument('--model',type=Path,action='append',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
sources={}
for d in ['neutrino','benchmarks','tests']:
    for f in (root/d).rglob('*'):
        if f.suffix in ('.py','.cpp','.c','.h','.cu','.cuh') and 'build' not in f.parts:
            sources[str(f.relative_to(root))]=sha(f)
files=[root/'neutrino/build/libcuda.so.1',root/'neutrino/build/preload.so',root/'neutrino/build/profile-preload.so',root/'benchmarks/capture_pack.cubin',root/'benchmarks/capture_pack_fast.cubin',root/'benchmarks/libqwen_dynamic.so',root/'benchmarks/deep_cupti_gpuonly.so',*a.model,*a.backend.glob('lib*.so')]
commands=[]
for cmd in [['nvidia-smi','-q'],['/usr/local/cuda/bin/nvcc','--version'],['uname','-a'],['free','-m'],['stat','-f','-c','%T','/dev/shm'],['/usr/sbin/nvpmodel','-q']]:
    try:
        r=subprocess.run(cmd,capture_output=True,text=True);commands.append(dict(argv=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    except FileNotFoundError:commands.append(dict(argv=cmd,unavailable=True))
cu=C.CDLL('libcuda.so.1');assert cu.cuInit(0)==0;dev=C.c_int();assert cu.cuDeviceGet(C.byref(dev),0)==0
attrs={}
for name,enum in [('sm_count',16),('integrated',18),('compute_major',75),('compute_minor',76),('clock_rate_khz',13)]:
    v=C.c_int();assert cu.cuDeviceGetAttribute(C.byref(v),enum,dev)==0;attrs[name]=v.value
left=list(Path('/dev/shm/neutrino-cross-device-20261004').rglob('*'))
data=dict(platform=platform.platform(),python=sys.version,root=str(root),sources=sources,files_sha256={str(f):sha(f) for f in files if f.is_file()},commands=commands,cuda_attributes=attrs,trace_leftovers=[str(f) for f in left],clocks_locked=False,llama_revision='unverified binary; actual libraries hashed')
(root/'results/provenance.json').write_text(json.dumps(data,indent=2)+'\n')
assert not left,'Raw trace cleanup incomplete'
