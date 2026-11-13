"""Combine real remote runtime hashes with final local source/evidence audit."""
from pathlib import Path
import json,hashlib,datetime
r=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
remote=json.loads((r/'remote-provenance.json').read_text())
assert json.loads((r/'correctness-audit.json').read_text())['complete']
local=[];retention=[]
for d in sorted((r/'capture').glob('r[1-7]-*')):
 if not (d/'retention.json').exists():continue
 for entry in json.loads((d/'retention.json').read_text()):
  assert entry['roundtrip_exact']
  if entry['retained_host']=='local':
   p=Path(entry['retained_path']);assert p.exists() and p.stat().st_size==entry['archive_bytes'];assert sha(p)==entry['archive_sha256']
   local.append({'condition':d.name,'path':str(p),'bytes':p.stat().st_size,'sha256':entry['archive_sha256'],'final_hash_verified':True})
  else:
   matches=[x for x in remote['retained_archive_audit'] if x.get('path')==entry['retained_path']]
   assert len(matches)==1 and matches[0]['final_hash_verified']
  retention.append({'condition':d.name,**entry})
assert len({x['condition'] for x in retention})==28
code={}
for d in ['code','source','diagnostic-source']:
 for p in sorted((r/d).rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
  rel=str(p.relative_to(r));code[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
# These are the timed logic and actual fine-probe generation inputs. Local
# report/provenance scripts need not have been uploaded to the execution host.
critical=['code/case_workload.py','code/case_analyze.py','code/reference_decode.cpp',
 'source/neutrino/src/cuda.c','source/neutrino/src/arena_capture.h',
 'source/benchmarks/nvcomp_pipeline.cu','source/benchmarks/llama_adapter_dynamic.cpp',
 'diagnostic-source/neutrino/src/cuda.c','diagnostic-source/neutrino/probe/cuda.py',
 'code/diagnostic_probe.py']
for name in critical:assert code[name]['sha256']==remote['case_files'][name]['sha256'],name
evidence={}
names=['allocation.json','execution-amendment.md','statistics.json','measurements.csv','correctness-audit.json',
 'fast-decoder-validation.json','batch-correctness.json','native-context.json','address-diagnosis.json',
 'consumer-profile/profile.json','diagnostic-v7/retention.json','diagnostic-v7/output-validation.json','frozen-verification.json',
 'figures/figure-manifest.json']
names += [f'figures/memory-layout-{token}.{ext}' for token in [0,1,63] for ext in ['png','pdf']]
for name in names:
 p=r/name;evidence[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_commit':None,
 'version_method':'Measured binaries/model/SDK and isolated code fixed by SHA-256; no Git commit fabricated.',
 'remote_runtime_inventory':'remote-provenance.json','remote_runtime_inventory_sha256':sha(r/'remote-provenance.json'),
 'local_code':code,'critical_local_remote_sources_identical':critical,'evidence':evidence,
 'formal_retention':retention,'local_archive_final_hash_audit':local,
 'formal_processes':28,'frozen_evidence_untouched':json.loads((r/'frozen-verification.json').read_text())['passed']}
(r/'final-provenance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'formal_processes':28,'local_archives_final_hash_verified':len(local),'critical_sources_identical':len(critical)}))
