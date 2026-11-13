"""Read-only remote source/binary inventory, plus exact retained-archive audit.

Run only after the formal matrix has finished. Writes new metadata solely to
this isolated case directory; never visits the frozen paper evidence.
"""
from pathlib import Path
import json,hashlib,subprocess,platform,sys,datetime
r=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def record(p):
 p=Path(p);q=p.resolve(strict=True)
 return {'path':str(p),'resolved_path':str(q),'bytes':q.stat().st_size,'sha256':sha(q)}
for n in range(1,8):
 for c in ['serial_neutrino','async_raw','nvcomp_cascaded','ispa_ere']:
  assert (r/'capture'/f'r{n}-{c}'/'complete.json').exists()
files={}
for d in ['source','diagnostic-source','code']:
 for p in sorted((r/d).rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.log']:continue
  files[str(p.relative_to(r))]=record(p)
sdk=r/'deps/nvcomp-linux-x86_64-5.3.0.16_cuda12-archive'
paths=[
 '/root/autodl-tmp/neutrino-review-sm-only-20261009/models-official/Qwen3-8B-Q4_K_M.gguf',
 '/root/autodl-tmp/neutrino-review-sm-only-20261009/llama-jetson-build/bin/libllama.so',
 '/root/autodl-tmp/neutrino-review-sm-only-20261009/llama-jetson-build/bin/libggml.so',
 '/root/autodl-tmp/neutrino-review-sm-only-20261009/llama-jetson-build/bin/libggml-cuda.so',
 '/root/autodl-tmp/ispa-3090new-cpu-build/bin/libggml-base.so',
 '/root/autodl-tmp/ispa-3090new-cpu-build/bin/libggml-cpu.so',
 '/root/autodl-tmp/neutrino-inference-case-20261013/backend/libggml-cpu.so',
 str(sdk/'lib/libnvcomp.so.5.3.0.16'),
 str(sdk/'lib/libnvcomp_cpu.so.5.3.0.16'),
 '/usr/lib/x86_64-linux-gnu/libcuda.so.570.124.04',
 '/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
]
loaded=json.loads((r/'loaded-library-paths.json').read_text())
paths += [p for process in loaded['processes'] for p in process['mapped_library_paths']]
external=[record(p) for p in sorted(set(paths))]
retained=[]
for d in sorted((r/'capture').glob('r[1-7]-*')):
 if d.name.endswith('failed-config'):continue
 for entry in json.loads((d/'archives.json').read_text()):
  p=Path(entry['archive'])
  if p.exists():
   assert p.stat().st_size==entry['archive_bytes']
   assert sha(p)==entry['archive_sha256']
   retained.append({'condition':d.name,'path':str(p),'host':'3090-new','bytes':p.stat().st_size,'sha256':entry['archive_sha256'],'final_hash_verified':True})
  else:
   retained.append({'condition':d.name,'original_remote_path':str(p),'host':'local','note':'Remote duplicate removed only after controller verified local archive SHA; final local check recorded separately.'})
diag=json.loads((r/'diagnostic-v7/retention.json').read_text());p=Path(diag['archive']);assert p.exists() and p.stat().st_size==diag['archive_bytes'];assert sha(p)==diag['archive_sha256']
retained.append({'condition':'diagnostic-v7','path':str(p),'host':'3090-new','bytes':p.stat().st_size,'sha256':diag['archive_sha256'],'final_hash_verified':True,'every_original_file_roundtrip_exact':diag['roundtrip_every_file_exact']})
commands={}
for name,cmd in [('nvcc',['/usr/local/cuda/bin/nvcc','--version']),('cxx',['g++','--version']),('gpu',['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),('disk',['df','-B1','/root/autodl-tmp','/root','/dev/shm'])]:
 commands[name]=subprocess.check_output(cmd,text=True).strip()
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'3090-new','python':sys.version,'platform':platform.platform(),'tools':commands,'case_files':files,'external_runtime_and_model':external,'observed_loaded_libraries':loaded,'retained_archive_audit':retained,'notes':['GPU binary reused unchanged; source tree serves explanatory context, binary hash is the measured version. No Git commit invented.','Actual /proc process maps from r7 nvCOMP identify the compatible CPU/base libraries and common GPU libraries. Candidate backend symlinks are also recorded.','Archive final SHA audit is after all formal timing and cannot affect GPU comparisons.']}
(r/'remote-provenance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'files':len(files),'external':len(external),'archives':len(retained),'all_remote_archive_hashes_exact':True}))
