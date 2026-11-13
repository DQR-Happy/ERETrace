"""Analyze and export compact evidence after both suites complete."""
import hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/neutrino-b16-attribution-20261007')
for name in ['status.json','gpu-status.json']:
    while True:
        try:s=json.loads((ROOT/name).read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
        if any(m['status']=='failed' for m in s['matrices']):raise RuntimeError(f'{name} failed')
        if s['complete']:break
        time.sleep(10)
subprocess.run([sys.executable,'-B',str(ROOT/'analyze.py')],check=True)
hashes={}
for p in (ROOT/'collector-old/neutrino/src').glob('*'):
    if p.suffix in ['.c','.h']:hashes[str(p.relative_to(ROOT/'collector-old'))]=hashlib.sha256(p.read_bytes()).hexdigest()
for rel in ['neutrino/build/libcuda.so.1','benchmarks/capture_pack.cubin']:
    hashes[rel]=hashlib.sha256((ROOT/'collector-old'/rel).read_bytes()).hexdigest()
(ROOT/'collector-file-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
collector=Path('/root/autodl-tmp/neutrino-paper-eval-20261005/regression-recheck-20261005/deep_gpuonly.so')
(ROOT/'gpu-collector-provenance.json').write_text(json.dumps(dict(path=str(collector),sha256=hashlib.sha256(collector.read_bytes()).hexdigest(),callbacks=False,activity=True),indent=2)+'\n')
assert not Path('/dev/shm/neutrino-b16-attribution-20261007').exists()
assert not Path('/dev/shm/neutrino-b16-gpu-20261007').exists()
entries=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if rel.parts[0]=='collector-old' or '__pycache__' in rel.parts or p.name=='export-manifest.json':continue
    if p.suffix not in ['.json','.jsonl','.py','.c','.h','.cpp','.log','.md','.csv','.txt','.png','.pdf','.svg']:continue
    entries.append((p,rel.as_posix()))
manifest={r:hashlib.sha256(p.read_bytes()).hexdigest() for p,r in entries}
(ROOT/'export-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=ROOT.parent/'neutrino-b16-attribution-20261007-evidence.tar'
with tarfile.open(archive,'w') as tf:
    for p,r in entries:tf.add(p,arcname=r)
    tf.add(ROOT/'export-manifest.json',arcname='export-manifest.json')
print(json.dumps(dict(archive=str(archive),files=len(entries),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())),flush=True)
print('FINAL EXPORT COMPLETE',flush=True)
