"""Retain source/binary hashes and disassembly, never trace or model bytes."""
import argparse,hashlib,json,platform,re,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
sources={str(p.relative_to(root)):sha(p) for d in ['neutrino','benchmarks','reference'] for p in (root/d).rglob('*') if p.suffix in ('.py','.c','.h','.cu','.cpp') and '__pycache__' not in p.parts}
binaries={str(p.relative_to(root)):sha(p) for d in ['neutrino/build','benchmarks','results/core-micro/build'] for p in (root/d).glob('*') if p.suffix in ('.so','.cubin') or p.name=='libcuda.so.1'}
if a.model:binaries[str(a.model)]=sha(a.model)
commands=[]
for cmd in [['uname','-a'],['nvidia-smi','-q'],['/usr/local/cuda/bin/nvcc','--version'],['stat','-f','-c','%T','/dev/shm']]:
    try:r=subprocess.run(cmd,text=True,capture_output=True);commands.append(dict(argv=cmd,rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
    except FileNotFoundError:commands.append(dict(argv=cmd,missing=True))
rows=[]
for manifest in (root/'compiler-cache').glob('*/manifest.json'):
    m=json.loads(manifest.read_text());mode=m['key_data']['env'].get('NEUTRINO_CORE_MODE','');name=m['key_data']['kernel'];dest=root/'results/sass'/mode;dest.mkdir(parents=True,exist_ok=True)
    for version in ['pruned','probed']:
        binary=manifest.parent/(version+'.bin')
        sass=subprocess.check_output(['cuobjdump','--dump-resource-usage','--dump-sass',str(binary)],text=True)
        file=dest/(manifest.parent.name[:16]+'-'+version+'.txt');file.write_text(sass)
        ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@[^\s]+\s+)?([A-Z][\w.]*)([^;]*);',sass)
        resource=re.search(r'REG:(\d+).*?SHARED:(\d+).*?LOCAL:(\d+)',sass)
        rows.append(dict(mode=mode,kernel=name,version=version,binary_sha=sha(binary),sass_file=str(file.relative_to(root/'results')),instructions=len(ops),stg=sum(op.startswith('STG') for op,args in ops),mufu=sum(op.startswith('MUFU') for op,args in ops),clock_reads=sum('SR_CLOCK' in args for op,args in ops),registers=int(resource[1]) if resource else None,shared=int(resource[2]) if resource else None,local=int(resource[3]) if resource else None,opcode_histogram={op:sum(x==op for x,y in ops) for op in sorted({x for x,y in ops})}))
(root/'results/compiler-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
left=[str(x) for x in Path('/dev/shm/neutrino-core-20261004').rglob('*')]
(root/'results/provenance.json').write_text(json.dumps(dict(root=str(root),python=sys.version,platform=platform.platform(),sources=sources,binaries=binaries,commands=commands,trace_leftovers=left,clocks_locked=False),indent=2)+'\n')
assert not left,'trace cleanup incomplete'
