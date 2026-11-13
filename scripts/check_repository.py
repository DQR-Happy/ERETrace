#!/usr/bin/env python3
"""Validate public payload, local links, dependency boundary and file integrity."""
import argparse, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'provenance/RELEASE-MANIFEST.json'

def files():
    runtime=json.loads((ROOT/'provenance/upstream-runtime.json').read_text())['files']
    skip={'.git','__pycache__','outputs','external','build','dist','.venv','verification','models'}
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
                  and not set(p.relative_to(ROOT).parts)&skip
                  and not any(x.endswith('.egg-info') for x in p.relative_to(ROOT).parts)
                  and str(p.relative_to(ROOT)) not in runtime and p.name!='.DS_Store'
                  and p.suffix not in {'.pyc','.so','.cubin','.log'})

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check(write=False):
    bad=[];payload=files();runtime=json.loads((ROOT/'provenance/upstream-runtime.json').read_text())['files']
    patterns=[rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{60,}',rb'AKIA[0-9A-Z]{16}']
    for p in payload:
        rel=str(p.relative_to(ROOT));data=p.read_bytes()
        if p.name in {'.env','id_rsa','id_ed25519'} or p.suffix in {'.gguf','.safetensors','.pem','.key','.bin','.nts','.ntp','.ntz'}:bad.append('Forbidden payload: '+rel)
        if len(data)>50*1024**2:bad.append('Oversized payload: '+rel)
        if any(re.search(v,data) for v in patterns):bad.append('Credential pattern: '+rel)
        if p.suffix=='.md':
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',data.decode()):
                if '://' in link or link.startswith(('#','mailto:')):continue
                target=link.split('#')[0]
                if target and not (p.parent/target).exists():bad.append('Broken link '+rel+': '+target)
    for name,v in runtime.items():
        p=ROOT/name
        if not p.is_file() or digest(p)!=v['sha256']:bad.append('Runtime missing or modified: '+name)
    entries={str(p.relative_to(ROOT)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in payload if p!=MANIFEST}
    if write:
        MANIFEST.write_text(json.dumps({'version':'1.0.0','files':entries},indent=2)+'\n')
    elif MANIFEST.exists():
        if entries!=json.loads(MANIFEST.read_text())['files']:bad.append('Release manifest differs; inspect changes before regenerating')
    else:bad.append('Release manifest missing')
    if bad:raise SystemExit('\n'.join(bad))
    print(json.dumps({'passed':True,'public_files':len(entries)+1,'runtime_files':len(runtime),'payload_bytes':sum(v['bytes'] for v in entries.values())},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-manifest',action='store_true');check(parser.parse_args().write_manifest)
