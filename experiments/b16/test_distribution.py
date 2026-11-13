"""Synthetic boundary checks for the offline statistics reader (CPU only)."""
import json,struct,tempfile
from pathlib import Path
from trace_distribution import collect
root=Path(__file__).resolve().parents[1]
results=[]
with tempfile.TemporaryDirectory(dir=root) as td:
    td=Path(td)
    for ew in [2,3,4]:
        n=12;groups=3;m=9;localbytes=(m*12+7)//8
        values=[0,1,2,3,32768,65535,65536,131072,100000,42,256,(1<<(8*ew))-1]
        values=[min(v,(1<<(8*ew))-1) for v in values]
        payload=bytearray(2064+4*groups+localbytes+n*ew)
        struct.pack_into('<4I',payload,0,0,4,ew,5)
        start=2064+4*groups+localbytes
        for i,v in enumerate(values):payload[start+i*ew:start+(i+1)*ew]=v.to_bytes(ew,'little')
        magic=f'NTRF02{ew}\n'.encode()
        hdr=struct.pack('<10IQ',3,1,1,128,1,1,0,1,16,32,48)
        path=td/f'ew{ew}.nts';path.write_bytes(magic+hdr+struct.pack('<Q',(3<<62)|len(payload))+payload)
        dest=td/f'ew{ew}.jsonl';collect(path,dest,'synthetic')
        got=json.loads(dest.read_text())['buckets'][0]
        assert got['elapsed_max']==max(values) and got['elapsed_gt65535']==sum(v>65535 for v in values)
        hist=[0]*33
        for v in values:hist[v.bit_length()]+=1
        assert got['elapsed_bit_hist']==hist and got['records']==n
        results.append(dict(elapsed_width=ew,passed=True,records=n))
    n=2;payload=struct.pack('<QIIQII',2**64-1,2**32-1,81,0,65535,0)
    hdr=struct.pack('<10IQ',1,1,1,64,1,1,0,1,16,32,48)
    path=td/'fallback.nts';path.write_bytes(b'NTRF022\n'+hdr+struct.pack('<Q',(1<<62)|32)+payload)
    dest=td/'fallback.jsonl';collect(path,dest,'synthetic')
    got=json.loads(dest.read_text())['buckets'][0]
    assert got['tag']==1 and got['elapsed_max']==2**32-1 and got['elapsed_gt65535']==1
    results.append(dict(raw_fallback=True,passed=True,records=n))
    packet=hdr+struct.pack('<Q',(1<<62)|32)+payload
    phase=td/'phase.nts';phase.write_bytes(b'NTRF022\n'+packet+packet)
    phase.with_suffix('.ids').write_text('0,0,8,32,32\n1,1,96,32,32\n')
    dest=td/'phase.jsonl';collect(phase,dest,'synthetic')
    buckets=json.loads(dest.read_text())['buckets']
    assert {b['token_phase'] for b in buckets}=={'prefill','decode'} and all(b['frames']==1 for b in buckets)
    results.append(dict(token_phase=True,passed=True,frames=2))
(root/'statistics-reader-test.json').write_text(json.dumps(dict(passed=True,cases=results),indent=2)+'\n')
print('5 offline statistics boundary cases passed')
