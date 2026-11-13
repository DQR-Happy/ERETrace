"""Exact full-request archive; explicit disposition of invalid exploratory runs."""
from pathlib import Path
import json,hashlib,tarfile,shutil
import zstandard as z
r=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
# These runs have invalid branch tags, and are never formal performance data.
# Retain all generated code and the three complete diagnostic files, already
# copied out by extract_diagnostic, before freeing invalid transient captures.
for version in [3,4,5]:
 src=Path(f'/dev/shm/neutrino-case-diagnostic-v{version}');dst=r/f'diagnostic-v{version}'
 if not src.exists():continue
 runtime=dst/'invalid-runtime-code';runtime.mkdir(exist_ok=True)
 manifest=[]
 for p in src.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(src);manifest.append({'path':str(rel),'bytes':p.stat().st_size})
  if 'result' not in rel.parts:
   target=runtime/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 (dst/'disposition.json').write_text(json.dumps({'status':'invalid exploratory diagnostic; no address or performance claims','reason':'uninitialized branch tag; literal assignment omitted by DSL in v5; address lifetime additionally fixed in v4','retained':'three full detailed Map files, extracted arrays, generated code and logs','invalid_transient_files':manifest},indent=2)+'\n')
 shutil.rmtree(src)
src=Path('/dev/shm/neutrino-case-diagnostic-v7');dst=r/'diagnostic-v7'
assert src.exists() and (dst/'extracted/summary.json').exists()
files=sorted(p for p in src.rglob('*') if p.is_file())
manifest=[{'path':str(p.relative_to(src)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in files]
(dst/'full-file-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=dst/'full-request.tar.zst'
with archive.open('wb') as f:
 with z.ZstdCompressor(level=5,threads=4).stream_writer(f) as stream:
  with tarfile.open(fileobj=stream,mode='w|',format=tarfile.PAX_FORMAT) as tar:
   for p in files:tar.add(p,arcname=str(p.relative_to(src)),recursive=False)
expected={x['path']:x for x in manifest};seen=set()
with archive.open('rb') as f,z.ZstdDecompressor().stream_reader(f) as stream:
 with tarfile.open(fileobj=stream,mode='r|') as tar:
  for entry in tar:
   assert entry.isfile() and entry.name in expected
   h=hashlib.sha256();n=0;file=tar.extractfile(entry)
   for b in iter(lambda:file.read(8<<20),b''):h.update(b);n+=len(b)
   assert n==expected[entry.name]['bytes'] and h.hexdigest()==expected[entry.name]['sha256'];seen.add(entry.name)
assert seen==set(expected)
proof={'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'files':len(files),'raw_bytes':sum(x['bytes'] for x in manifest),'roundtrip_every_file_exact':True,'retained_host':'3090-new'}
(dst/'retention.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
shutil.rmtree(src)
