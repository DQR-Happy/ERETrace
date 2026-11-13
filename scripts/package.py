#!/usr/bin/env python3
"""Package the public source/patch payload, excluding materialized dependencies."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from check_repository import ROOT,check,files

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out==ROOT or ROOT in out.parents:p.error('Write release assets outside this checkout')
    if out.exists():p.error('Refusing to overwrite an existing release asset')
    check();out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(out,'w:gz') as tar:
        for f in files():
            info=tar.gettarinfo(str(f),'ERETrace-1.0.0/'+str(f.relative_to(ROOT)));info.uid=info.gid=0;info.uname=info.gname='';info.mtime=1794614400
            with f.open('rb') as stream:tar.addfile(info,stream)
    h=hashlib.sha256(out.read_bytes()).hexdigest();out.with_suffix(out.suffix+'.sha256').write_text(h+'  '+out.name+'\n')
    print(json.dumps({'archive':out.name,'bytes':out.stat().st_size,'sha256':h},indent=2))
if __name__=='__main__':main()
