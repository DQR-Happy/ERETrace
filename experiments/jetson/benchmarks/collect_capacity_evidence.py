"""Read-only device identity and generated capacity benchmark artifact hashes."""
import ctypes as C
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cuda = C.CDLL('libcuda.so.1')
def checked(code):
    if code:
        raise RuntimeError(f'CUDA query failed: {code}')
checked(cuda.cuInit(0))
dev = C.c_int()
checked(cuda.cuDeviceGet(C.byref(dev), 0))
name = C.create_string_buffer(256)
checked(cuda.cuDeviceGetName(name, 256, dev))
attrs = {}
for key, enum in [('sm_count', 16), ('compute_major', 75), ('compute_minor', 76)]:
    value = C.c_int()
    checked(cuda.cuDeviceGetAttribute(C.byref(value), enum, dev))
    attrs[key] = value.value
driver = C.c_int()
checked(cuda.cuDriverGetVersion(C.byref(driver)))
artifacts = {}
static = {}
for directory in sorted((root/'results').glob('capacity-*')):
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix in ('.ptx', '.cubin', '.so', '.log'):
            artifacts[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            if directory.name=='capacity-main' and path.suffix=='.cubin':
                sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],text=True,capture_output=True,check=True).stdout
                opcodes=re.findall(r'^\s*/\*[0-9a-fA-F]+\*/\s*(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)(?=[\s;])',sass,re.M)
                if not opcodes:raise ValueError('no SASS instructions found')
                static[path.stem]=dict(instructions=len(opcodes),opcodes=dict(Counter(opcodes)),sass_sha256=hashlib.sha256(sass.encode()).hexdigest())
out = dict(device=name.value.decode(), driver_api_version=driver.value,
           **attrs, artifact_sha256=artifacts, static_sass=static,
           note='Identity queried after runs. No claim of clocks or temperatures during runs.')
(root/'results/capacity-provenance.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('artifact_sha256','static_sass')}))
