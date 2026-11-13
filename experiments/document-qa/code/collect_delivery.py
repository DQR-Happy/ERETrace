"""Archive metadata after all performance runs; reference full traces separately."""
from pathlib import Path
import hashlib,json,sys,tarfile,platform
import zstandard as z
r=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
ret=[]
for p in sorted((r/'capture').glob('*/archives.json')):
 for x in json.loads(p.read_text()):
  target=Path(x['archive']);assert target.exists() and target.stat().st_size==x['archive_bytes']
  ret.append({'run':p.parent.name,'host':'3090-new','retained_path':str(target),'storage':'volatile tmpfs (retain while instance runs)' if str(target).startswith('/dev/shm') else 'instance filesystem (retain while instance exists)',**x})
assert len(ret)==16 and all(x['roundtrip_exact'] for x in ret)
(r/'retention-index.json').write_text(json.dumps({'formal_traces':len(ret),'all_full_traces_present':True,'total_archive_bytes':sum(x['archive_bytes']for x in ret),'volatile_archive_bytes':sum(x['archive_bytes']for x in ret if str(x['archive']).startswith('/dev/shm')),'files':ret},indent=2)+'\n')
# Verify measured versions did not change while experiment ran.
m=json.loads((r/'manifest.json').read_text())
assert all(sha(Path(x['path']))==x['sha256']for x in m['files'])
(r/'measured-version-audit.json').write_text(json.dumps({'all_preregistered_file_hashes_unchanged':True,'files':len(m['files'])},indent=2)+'\n')
metadata=r/'remote-metadata.tar.zst'
with metadata.open('wb') as f,z.ZstdCompressor(level=5,threads=4).stream_writer(f) as stream,tarfile.open(fileobj=stream,mode='w|') as t:
 for p in sorted((r/'capture').rglob('*')):
  if p.is_file():t.add(p,arcname=str(p.relative_to(r)),recursive=False)
 t.add(r/'execution.log',arcname='execution.log')
manifest=[]
for p in sorted(r.glob('*')):
 if p.is_file():manifest.append({'relative':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
for p in sorted((r/'code').glob('*')):
 if p.is_file():manifest.append({'relative':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':sha(p),'resolved':str(p.resolve())})
(r/'delivery-manifest.json').write_text(json.dumps({'files':manifest,'full_trace_storage_note':'16 complete exact archives retained on 3090-new; tmpfs paths volatile; no instance restart or cleanup done','frozen_evidence_and_paper_untouched':True},indent=2)+'\n')
print(json.dumps({'metadata_archive_bytes':metadata.stat().st_size,'full_trace_archives':len(ret),'archive_bytes':sum(x['archive_bytes']for x in ret)}))
