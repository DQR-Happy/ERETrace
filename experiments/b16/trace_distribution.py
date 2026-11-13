"""Offline shape/representation/elapsed statistics; never retained raw records."""
import collections, json, struct
import numpy as np

def collect(path,destination,label):
    buckets={}
    ids=path.with_suffix('.ids')
    tokens=[int(line.split(',',2)[1]) for line in ids.read_text().splitlines()] if ids.exists() else None
    ordinal=0
    with path.open('rb') as fp:
        magic=fp.read(8)
        while True:
            hdr=fp.read(48)
            if not hdr:break
            assert len(hdr)==48
            gx,gy,gz,bx,by,bz,shared,maps,item,div,offset=struct.unpack('<10IQ',hdr)
            assert (maps,item,div,offset)==(1,16,32,48)
            n=gx*gy*gz*((bx*by*bz+31)//32);raw=16*n
            packed,=struct.unpack('<Q',fp.read(8));tag=packed>>62;size=packed&((1<<62)-1);stored=size or raw
            phase=('prefill' if tokens[ordinal]==0 else 'decode') if tokens is not None else 'unspecified'
            ordinal+=1
            key=(gx,gy,gz,bx,by,bz,shared,tag,phase)
            r=buckets.setdefault(key,dict(shape=list(key[:-2]),tag=tag,token_phase=phase,frames=0,records=0,raw_bytes=0,file_payload_bytes=0,elapsed_records_checked=0,elapsed_gt65535=0,elapsed_max=0,elapsed_bit_hist=[0]*33))
            r['frames']+=1;r['records']+=n;r['raw_bytes']+=raw;r['file_payload_bytes']+=stored
            elapsed=None
            if tag in (0,1):
                p=fp.read(stored);assert len(p)==stored
                elapsed=np.ndarray((n,),dtype='<u4',buffer=p,offset=8,strides=(16,))
            elif magic[:4]==b'NTRF':
                p=fp.read(stored);assert len(p)==stored
                flags,wp,ew,version=struct.unpack_from('<4I',p)
                assert flags==0 and ew in (2,3,4)
                if version>=2:
                    groups=(n+wp-1)//wp;m=n-groups if version==5 else n
                    bits=12 if version in (4,5) else 16
                    at=2064+4*groups+(m*bits+7)//8
                else:at=2064+n*3
                if ew in (2,4):elapsed=np.frombuffer(p,dtype='<u2' if ew==2 else '<u4',count=n,offset=at)
                else:
                    q=np.frombuffer(p,dtype=np.uint8,count=n*3,offset=at).reshape(n,3).astype(np.uint32)
                    elapsed=q[:,0]|q[:,1]<<8|q[:,2]<<16
            else:fp.seek(stored,1)
            if elapsed is not None:
                r['elapsed_records_checked']+=n;r['elapsed_gt65535']+=int(np.count_nonzero(elapsed>65535))
                r['elapsed_max']=max(r['elapsed_max'],int(elapsed.max(initial=0)))
                # Exact integer boundaries; no logarithmic rounding at powers of two.
                boundaries=np.array([0]+[(1<<b)-1 for b in range(1,33)],dtype=np.uint64)
                h=np.bincount(np.searchsorted(boundaries,elapsed,side='left'),minlength=33)
                for i,v in enumerate(h):r['elapsed_bit_hist'][i]+=int(v)
            assert fp.tell()<=path.stat().st_size
    if tokens is not None:assert ordinal==len(tokens)
    result=dict(request=label,format=magic.decode().strip(),frames=sum(r['frames'] for r in buckets.values()),records=sum(r['records'] for r in buckets.values()),buckets=list(buckets.values()))
    with destination.open('a') as fp:fp.write(json.dumps(result,separators=(',',':'))+'\n')
