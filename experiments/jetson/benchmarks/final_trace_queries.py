"""Validate actual analysis queries against same-Map, independent C raw oracles.

Only aggregate integer observations leave tmpfs. These are instrumented warp
cycle observations, not native GPU time or a cross-SM timeline.
"""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from trace_codec import decode_flex, decode_gpu9, decode_nvcomp

DTYPE = np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')])

def queries(raw):
    a = np.frombuffer(raw,dtype=DTYPE)
    sm = a['sm'].astype('int64')
    assert len(a) and int(sm.max())<256
    counts = np.bincount(sm,minlength=256)
    sums = np.zeros(256,dtype='uint64')
    np.add.at(sums,sm,a['elapsed'].astype('uint64'))
    # frexp's integer exponent is bit_length for exact uint32 inputs, zero -> 0.
    hist = np.bincount(np.frexp(a['elapsed'].astype('float64'))[1],minlength=33)
    return dict(records=len(a),cycle_sum=int(a['elapsed'].sum(dtype='uint64')),
        cycle_max=int(a['elapsed'].max()),log2_hist=list(map(int,hist)),
        per_sm={str(i):[int(counts[i]),int(sums[i])] for i in np.flatnonzero(counts)})

def oracle(line):
    head,hist,sm = line.strip().split(';')
    launch,token,n,total,maximum = map(int,head.split(','))
    per_sm = {}
    for entry in sm.rstrip(',').split(','):
        i,count,s = map(int,entry.split(':'));per_sm[str(i)]=[count,s]
    return launch,token,dict(records=n,cycle_sum=total,cycle_max=maximum,
        log2_hist=list(map(int,hist.split(','))),per_sm=per_sm)

def frames(path):
    with path.open('rb') as f:
        magic=f.read(8)
        assert magic[:4] in (b'NTGP',b'NTFX',b'NTHR',b'NTRF',b'NTNV')
        while True:
            offset=f.tell();h=f.read(48)
            if not h:break
            assert len(h)==48
            dims=struct.unpack('<10IQ',h)
            assert dims[7:]==(1,16,32,48)
            n=np.prod(dims[:3],dtype='int64')*((np.prod(dims[3:6],dtype='int64')+31)//32)
            n=int(n);packed,=struct.unpack('<Q',f.read(8));tag=packed>>62;size=packed&((1<<62)-1)
            data=f.read(size or n*16);assert len(data)==(size or n*16)
            if tag==3:data=decode_nvcomp(data,n,magic==b'NTNVCS1\n') if magic[:4]==b'NTNV' else decode_flex(data,n)
            elif tag==2:data=decode_gpu9(data,n)
            elif tag==0 and size:
                a=np.empty(n,dtype=DTYPE);a['start']=np.frombuffer(data[:n*8],dtype='<u8')
                a['elapsed']=np.frombuffer(data[n*8:n*12],dtype='<u4');a['sm']=np.frombuffer(data[n*12:],dtype='u1');data=a.tobytes()
            else:assert (tag==0 and size==0) or (tag==1 and size==n*16)
            yield offset,dims,data

def analyze_files(paths):
    files=[]
    for path in sorted(paths):
        path=Path(path);rawq=path.with_suffix('.rawq');ids=path.with_suffix('.ids')
        assert rawq.exists() and ids.exists(),str(path)
        rows=[];hist=np.zeros(33,dtype='uint64');sm_counts=np.zeros(256,dtype='uint64')
        by_token={};digest=hashlib.sha256();record_count=0
        with rawq.open() as r,ids.open() as index:
            for offset,dims,data in frames(path):
                launch,token,expected=oracle(next(r))
                ident=list(map(int,next(index).strip().split(',')))
                assert ident[:3]==[launch,token,offset] and ident[3]==len(data)
                actual=queries(data)
                assert actual==expected,(path,launch,'query results differ from original Map')
                digest.update(json.dumps([launch,token,actual],sort_keys=True).encode())
                record_count+=actual['records'];hist+=np.array(actual['log2_hist'],dtype='uint64')
                for sm,(count,_) in actual['per_sm'].items():sm_counts[int(sm)]+=count
                t=by_token.setdefault(str(token),dict(frames=0,records=0,cycle_sum=0,cycle_max=0))
                t['frames']+=1;t['records']+=actual['records'];t['cycle_sum']+=actual['cycle_sum'];t['cycle_max']=max(t['cycle_max'],actual['cycle_max'])
                rows.append(dict(launch=launch,token=token,shape=list(dims[:7]),records=actual['records'],
                                 warp_cycle_sum=actual['cycle_sum'],max_warp_cycles=actual['cycle_max']))
            assert not r.read().strip() and not index.read().strip()
        # Counts, distribution, and ranking are deterministic functions of each
        # exact frame query; matching the oracle proves all three are preserved.
        files.append(dict(file=path.name,frames=len(rows),records=record_count,exact=True,
            query_sha256=digest.hexdigest(),log2_cycle_histogram=list(map(int,hist)),
            sm_record_counts={str(i):int(sm_counts[i]) for i in np.flatnonzero(sm_counts)},
            tokens=by_token,top20_by_max_observed_warp_cycles=sorted(rows,key=lambda r:(-r['max_warp_cycles'],r['launch']))[:20]))
    return dict(passed=True,files=files,total_frames=sum(f['frames'] for f in files),
        total_records=sum(f['records'] for f in files),
        semantics='Exact integer queries on the same instrumented Maps; no native timing truth claim.')

