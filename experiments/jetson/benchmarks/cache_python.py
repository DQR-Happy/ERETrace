#!/root/autodl-tmp/neutrino-repro/.venv/bin/python
"""Experimental compiler launcher. Only cold compilation uses this executable.

Key: complete original binary, kernel, probes/filter/dynamic settings, every
Python source in the package, cuobjdump/ptxas binaries, and hardware fingerprint.
Every restored artifact is SHA-256 checked. Capture payload is never cached.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    begin=time.perf_counter_ns()
    script=Path(sys.argv[1]).resolve()
    work=Path(sys.argv[2]).resolve()
    kernel=sys.argv[3]
    root=Path(os.environ['NEUTRINO_COMPILER_CACHE_ROOT']).resolve()
    root.mkdir(parents=True,exist_ok=True)
    hardware=json.loads((root/'hardware.json').read_text())
    package=script.parents[1]
    decision=None
    if os.environ.get('NEUTRINO_CORE_MODE')=='policy':
        sys.path.insert(0,str(package.parent))
        from neutrino.probe.core_policy import resolve_policy
        policy=json.loads(Path(os.environ['NEUTRINO_CORE_POLICY']).read_text())
        decision=resolve_policy(policy,kernel,sha(work/'original.bin'),hardware,
            {n:sha(package/'probe'/n) for n in ('cuda.py','core_codegen.py')},
            {n:sha(Path(shutil.which(n))) for n in ('ptxas','cuobjdump')})
        os.environ['NEUTRINO_CORE_MODE']=decision['mode']
    sources={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))
             if '__pycache__' not in p.parts and 'build' not in p.parts}
    key_data=dict(original=sha(work/'original.bin'),kernel=kernel,sources=sources,
                  hardware=hardware,tools={name:sha(Path(shutil.which(name))) for name in ('ptxas','cuobjdump')},
                  env={name:os.environ.get(name,'') for name in
                       ('NEUTRINO_PROBES','NEUTRINO_FILTER','NEUTRINO_KERNEL','NEUTRINO_DYNAMIC','NEUTRINO_CORE_MODE')})
    if decision:key_data['core_policy']=decision
    key=hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()
    dest=root/key
    hit=False
    if (dest/'manifest.json').exists():
        manifest=json.loads((dest/'manifest.json').read_text())
        if manifest['key_data']!=key_data or any(sha(dest/n)!=h for n,h in manifest['files'].items()):
            raise RuntimeError('Compiler cache integrity mismatch')
        for n in manifest['files']:
            shutil.copyfile(dest/n,work/n)
        hit=True
    else:
        subprocess.run([sys.executable,str(script),*sys.argv[2:]],check=True)
        names=['kernel.info','pruned.bin','probed.bin','pruned.ptx','probed.ptx','process.log']
        if os.environ.get('NEUTRINO_DYNAMIC'):
            names+=['countd.bin','countd.ptx']
        with tempfile.TemporaryDirectory(dir=root,prefix=key+'.') as tmp:
            tmp=Path(tmp)
            for n in names:
                shutil.copyfile(work/n,tmp/n)
            manifest=dict(key_data=key_data,files={n:sha(tmp/n) for n in names})
            (tmp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            # Benchmark suite is serial; no competing writer is expected.
            os.rename(tmp,dest)
    (work/'compile-cache.json').write_text(json.dumps(dict(key=key,hit=hit,
        elapsed_ns=time.perf_counter_ns()-begin,hardware=hardware))+'\n')
    if decision:(work/'core-policy.json').write_text(json.dumps(decision,indent=2)+'\n')


if __name__=='__main__':
    main()
