"""Common full-frame parser: same wire formats, math.prod for fixed dimensions."""
import math,struct
import numpy as np
from trace_codec import decode_flex,decode_gpu9,decode_nvcomp
DTYPE=np.dtype([("start","<u8"),("elapsed","<u4"),("sm","<u4")])

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
            n=math.prod(dims[:3])*((math.prod(dims[3:6])+31)//32)
            n=int(n);packed,=struct.unpack('<Q',f.read(8));tag=packed>>62;size=packed&((1<<62)-1)
            data=f.read(size or n*16);assert len(data)==(size or n*16)
            if tag==3:data=decode_nvcomp(data,n,magic==b'NTNVCS1\n') if magic[:4]==b'NTNV' else decode_flex(data,n)
            elif tag==2:data=decode_gpu9(data,n)
            elif tag==0 and size:
                a=np.empty(n,dtype=DTYPE);a['start']=np.frombuffer(data[:n*8],dtype='<u8')
                a['elapsed']=np.frombuffer(data[n*8:n*12],dtype='<u4');a['sm']=np.frombuffer(data[n*12:],dtype='u1');data=a.tobytes()
            else:assert (tag==0 and size==0) or (tag==1 and size==n*16)
            yield offset,dims,data
