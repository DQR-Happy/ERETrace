"""Compact actual-policy identities and build provenance; no trace retention."""
import hashlib,json,platform,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
applications=[]
for complete in sorted((root/'results').glob('policy-*/complete.json')):
    for runfile in sorted(complete.parent.glob('r*/run.json')):
        run=json.loads(runfile.read_text());entries=[]
        for item in json.loads((runfile.parent/'compile-cache.json').read_text()):
            m=json.loads((root/'compiler-cache'/item['key']/'manifest.json').read_text())
            k=m['key_data'];decision=k.get('core_policy')
            entries.append(dict(kernel=k['kernel'],selected_mode=k['env']['NEUTRINO_CORE_MODE'],decision=decision,binary_sha256=m['files']['probed.bin'],cache_key=item['key']))
        applications.append(dict(batch=complete.parent.name,round=run['round'],condition=run['condition'],entries=entries))
(root/'results/policy-application.json').write_text(json.dumps(applications,indent=2)+'\n')
sources={str(p.relative_to(root)):sha(p) for directory in ['neutrino','benchmarks'] for p in (root/directory).rglob('*') if p.suffix in ('.py','.c','.h','.cu','.cpp') and '__pycache__' not in p.parts}
binaries={str(p.relative_to(root)):sha(p) for directory in ['neutrino/build','benchmarks'] for p in (root/directory).glob('*') if p.suffix in ('.so','.cubin') or p.name=='libcuda.so.1'}
commands=[]
for cmd in [['uname','-a'],['nvidia-smi','-q'],['/usr/local/cuda/bin/nvcc','--version'],['stat','-f','-c','%T','/dev/shm'],['free','-b']]:
    try:
        r=subprocess.run(cmd,text=True,capture_output=True);commands.append(dict(argv=cmd,rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
    except FileNotFoundError:commands.append(dict(argv=cmd,missing=True))
left=[str(p) for p in Path('/dev/shm/neutrino-policy-20261004').rglob('*')]
assert not left
models={str(p):sha(p) for p in Path('/autodl-fs/data/qwen3-small-aiperf/models').glob('Qwen3-*.gguf') if p.name in ['Qwen3-0.6B-Q4_K_M.gguf','Qwen3-1.7B-Q4_K_M.gguf']}
(root/'results/provenance.json').write_text(json.dumps(dict(root=str(root),python=sys.version,platform=platform.platform(),sources=sources,binaries=binaries,models=models,commands=commands,trace_leftovers=left,clocks_locked=False),indent=2)+'\n')
print('policy application runs',len(applications),'raw trace leftovers',len(left))
