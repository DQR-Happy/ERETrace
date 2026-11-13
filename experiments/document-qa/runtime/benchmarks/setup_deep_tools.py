"""Download isolated official tools; do not modify host packages or settings."""
import ctypes as C
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

root=Path('/root/autodl-tmp/neutrino-deep-tools-20261003');root.mkdir(exist_ok=True)
def fetch(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response,path.open('wb') as fp:
            shutil.copyfileobj(response,fp)
    return hashlib.sha256(path.read_bytes()).hexdigest()

ubuntu='https://archive.ubuntu.com/ubuntu/pool/main/s/strace/'
html=urllib.request.urlopen(ubuntu,timeout=60).read().decode()
name=next(n for n in re.findall(r'href="([^"]+amd64.deb)"',html) if n.startswith('strace_5.16-0ubuntu3'))
sha=fetch(ubuntu+name,root/name)
subprocess.run(['dpkg-deb','-x',str(root/name),str(root/'strace')],check=True)
repo='https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/'
packages=gzip.decompress(urllib.request.urlopen(repo+'Packages.gz',timeout=60).read()).decode()
entries=[dict(line.split(': ',1) for line in b.splitlines() if ': ' in line and not line.startswith(' '))
         for b in packages.split('\n\n')]
entry=next(e for e in entries if e.get('Package')=='nsight-compute-2024.1.1')
deb=root/Path(entry['Filename']).name
ncu_sha=fetch(repo+entry['Filename'].lstrip('./'),deb)
assert ncu_sha==entry['SHA256']
subprocess.run(['dpkg-deb','-x',str(deb),str(root/'ncu')],check=True)
# Software-only perf clock probe, without changing sysctl/capabilities.
libc=C.CDLL(None,use_errno=True);attr=(C.c_ubyte*128)()
import struct
struct.pack_into('IIQQQQQ',attr,0,1,128,0,0,0,0,(1<<0)|(1<<5)|(1<<6))
fd=libc.syscall(298,C.byref(attr),0,-1,-1,0)
perf=dict(fd=fd,errno=C.get_errno())
if fd>=0:libc.close(fd)
data=dict(strace=dict(url=ubuntu+name,sha256=sha),ncu=dict(url=repo+entry['Filename'].lstrip('./'),sha256=ncu_sha),
          perf_probe=perf,capability_status=Path('/proc/self/status').read_text(),
          perf_event_paranoid=Path('/proc/sys/kernel/perf_event_paranoid').read_text(),
          nvidia_params=Path('/proc/driver/nvidia/params').read_text())
(root/'setup.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
