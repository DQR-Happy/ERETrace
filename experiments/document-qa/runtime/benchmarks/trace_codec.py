"""Read versioned lossless NTZ frames. Raw decoded trace is never archived."""
import ctypes as C
import hashlib
import struct


def record_stats(data):
    import numpy as np
    a=np.frombuffer(data,dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')]))
    assert not np.any(a['sm']==0xdeadbeef), 'Unwritten SM sentinel'
    assert not np.any(a['start']==0xdeadbeefdeadbeef), 'Unwritten clock sentinel'
    assert not np.any(a['elapsed']==0xdeadbeef), 'Unwritten elapsed sentinel in short-kernel test'
    zero=np.count_nonzero((a['start']==0)&(a['elapsed']==0)&(a['sm']==0))
    return len(a)-int(zero),int(zero),set(map(int,np.unique(a['sm'])))


def check_packed(path,records=False):
    import numpy as np
    frames=logical=nonzero=zeros=raw_frames=0;sm=set();digest=hashlib.sha256()
    with path.open('rb') as fp:
        magic=fp.read(8);assert magic in (b'NTPACK1\n',b'NTPACK2\n')
        while True:
            h=fp.read(48)
            if not h:break
            assert len(h)==48
            gx,gy,gz,bx,by,bz,shared,maps,item,div,offset=struct.unpack('<10IQ',h)
            assert min(gx,gy,gz,bx,by,bz)>0 and maps==1 and item==16 and div==32 and offset==48
            n=gx*gy*gz*((bx*by*bz+31)//32)
            if magic==b'NTPACK1\n':base,packed=struct.unpack('<QQ',fp.read(16))
            else:packed,=struct.unpack('<Q',fp.read(8))
            assert n*16<=9723904 and 0<=packed<n*16
            data=fp.read(packed or n*16);assert len(data)==(packed or n*16)
            if packed:
                bases=None
                if magic==b'NTPACK2\n':
                    count,=struct.unpack('<H',data[:2]);assert count<=255
                    bases=np.zeros(256,dtype='uint64');pos=2;seen=set()
                    for _ in range(count):
                        smid=data[pos];value,=struct.unpack('<Q',data[pos+1:pos+9]);assert smid<255 and smid not in seen
                        seen.add(smid);bases[smid]=value;pos+=9
                    data=data[pos:];assert len(data)==n*9
                a=np.frombuffer(data,dtype=np.dtype([('delta','<u4'),('elapsed','<u4'),('sm','u1')]))
                out=np.empty(n,dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')]))
                out['start']=a['delta'].astype('uint64')+(bases[a['sm']] if bases is not None else base)
                out['elapsed']=a['elapsed'];out['sm']=a['sm']
                if bases is not None:
                    assert all(int(i) in seen or i==255 for i in np.unique(a['sm']))
                    zero=a['sm']==255;assert not np.any(a['delta'][zero]) and not np.any(a['elapsed'][zero])
                    out[zero]=np.zeros(1,dtype=out.dtype)[0]
                data=out.tobytes()
            else:raw_frames+=1
            digest.update(h);digest.update(data)
            if records:
                nz,z,ids=record_stats(data);nonzero+=nz;zeros+=z;sm.update(ids)
            frames+=1;logical+=48+n*16
    return dict(file=path.name,bytes=path.stat().st_size,logical_bytes=logical,frames=frames,
                decoded_sha256=digest.hexdigest(),record_validation=records,nonzero_records=nonzero,
                zero_records=zeros,observed_sm_ids=sorted(sm),raw_fallback_frames=raw_frames,format=magic.decode().strip())


def check_gpu(path,records=False):
    import numpy as np
    frames=logical=nonzero=zeros=encoded_frames=0;sm=set();digest=hashlib.sha256()
    dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')])
    with path.open('rb') as fp:
        magic=fp.read(8);assert (magic in (b'NTGPU13\n',b'NTGPUV2\n',b'NTNVCA1\n',b'NTNVCS1\n') or ((magic[:4] in (b'NTFX',b'NTHR') and magic[6:]==b'!\n') or (magic[:4]==b'NTRF' and magic[7:]==b'\n')))
        while True:
            h=fp.read(48)
            if not h:break
            assert len(h)==48
            gx,gy,gz,bx,by,bz,shared,maps,item,div,offset=struct.unpack('<10IQ',h)
            assert min(gx,gy,gz,bx,by,bz)>0 and maps==1 and item==16 and div==32 and offset==48
            n=gx*gy*gz*((bx*by*bz+31)//32);packed,=struct.unpack('<Q',fp.read(8))
            tag=packed>>62;size=packed&((1<<62)-1)
            assert (magic[:4]==b'NTNV' and tag==3 and 16<size<n*48) or (tag==0 and size in (0,n*13)) or (magic==b'NTGPUV2\n' and ((tag==2 and size==n*9+2064) or (tag==1 and size==n*16))) or (magic[:4] in (b'NTFX',b'NTHR',b'NTRF') and ((tag==3 and size==gpu_format_size(magic,n,(bx*by*bz+31)//32)) or (tag==1 and size==n*16)))
            data=fp.read(size or n*16);assert len(data)==(size or n*16)
            if tag==3:
                data=decode_nvcomp(data,n,magic==b'NTNVCS1\n') if magic[:4]==b'NTNV' else decode_flex(data,n);encoded_frames+=1
            elif tag==2:
                data=decode_gpu9(data,n);encoded_frames+=1
            elif packed and tag!=1:
                out=np.empty(n,dtype=dtype)
                out['start']=np.frombuffer(data[:n*8],dtype='<u8')
                out['elapsed']=np.frombuffer(data[n*8:n*12],dtype='<u4')
                out['sm']=np.frombuffer(data[n*12:],dtype='u1');data=out.tobytes();encoded_frames+=1
            digest.update(h);digest.update(data)
            if records:
                nz,z,ids=record_stats(data);nonzero+=nz;zeros+=z;sm.update(ids)
            frames+=1;logical+=48+n*16
    return dict(file=path.name,bytes=path.stat().st_size,logical_bytes=logical,frames=frames,
                decoded_sha256=digest.hexdigest(),record_validation=records,nonzero_records=nonzero,
                zero_records=zeros,observed_sm_ids=sorted(sm),encoded_frames=encoded_frames,format=magic.decode().strip())


def check_compressed(path, records=False):
    lib=C.CDLL('libzstd.so.1')
    lib.ZSTD_decompress.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
    lib.ZSTD_decompress.restype=C.c_size_t
    lib.ZSTD_isError.argtypes=[C.c_size_t];lib.ZSTD_isError.restype=C.c_uint
    frames=logical=nonzero=zeros=0;sm=set();digest=hashlib.sha256()
    with path.open('rb') as fp:
        assert fp.read(8)==b'NTZSTD1\n'
        while True:
            h=fp.read(48)
            if not h:break
            assert len(h)==48
            gx,gy,gz,bx,by,bz,shared,maps,item,div,offset=struct.unpack('<10IQ',h)
            assert min(gx,gy,gz,bx,by,bz)>0 and maps==1 and item==16 and div==32 and offset==48
            raw_size=gx*gy*gz*((bx*by*bz+31)//32)*16
            packed,=struct.unpack('<Q',fp.read(8))
            assert raw_size<=9723904 and packed<=raw_size
            data=fp.read(packed or raw_size);assert len(data)==(packed or raw_size)
            if packed:
                dest=C.create_string_buffer(raw_size)
                n=lib.ZSTD_decompress(dest,raw_size,data,len(data))
                assert not lib.ZSTD_isError(n) and n==raw_size
                data=dest.raw
            digest.update(h);digest.update(data)
            if records:
                nz,z,ids=record_stats(data);nonzero+=nz;zeros+=z;sm.update(ids)
            frames+=1;logical+=48+raw_size
    return dict(file=path.name,bytes=path.stat().st_size,logical_bytes=logical,frames=frames,
                decoded_sha256=digest.hexdigest(),record_validation=records,nonzero_records=nonzero,
                zero_records=zeros,observed_sm_ids=sorted(sm),format='NTZSTD1')


def decode_gpu9(data,n):
    import numpy as np
    assert len(data)==2064+n*9 and struct.unpack_from('<I',data)[0]==0
    bases=np.frombuffer(data,offset=16,count=256,dtype='<u8')
    deltas=np.frombuffer(data,offset=2064,count=n,dtype='<i4')
    elapsed=np.frombuffer(data,offset=2064+4*n,count=n,dtype='<u4')
    sm=np.frombuffer(data,offset=2064+8*n,count=n,dtype='u1')
    out=np.empty(n,dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')]))
    out['start']=bases[sm]+deltas.astype('int64').astype('uint64')
    out['elapsed']=elapsed;out['sm']=sm
    return out.tobytes()


def decode_flex(data,n):
    import numpy as np
    flags,dw,ew,version=struct.unpack_from('<4I',data)
    if version==2:return decode_hier(data,n)
    if version in (3,4,5):return decode_reference(data,n)
    assert flags==0 and version==1 and dw in (2,3,4) and ew in (2,3,4)
    assert len(data)==2064+n*(dw+ew+1)
    def column(offset,width):
        a=np.frombuffer(data,offset=offset,count=n*width,dtype='u1').reshape(n,width)
        out=np.zeros(n,dtype='uint32')
        for k in range(width):out|=a[:,k].astype('uint32')<<(k*8)
        return out
    bits=column(2064,dw).astype('int64');sign=1<<(dw*8-1)
    deltas=(bits^sign)-sign
    sm=np.frombuffer(data,offset=2064+n*(dw+ew),count=n,dtype='u1')
    bases=np.frombuffer(data,offset=16,count=256,dtype='<u8')
    out=np.empty(n,dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')]))
    out['start']=bases[sm]+deltas.astype('uint64');out['elapsed']=column(2064+n*dw,ew);out['sm']=sm
    return out.tobytes()


def gpu_format_size(magic,n,wp):
    if magic[:4]==b'NTRF':
        lg,packed,ew=map(lambda x:int(chr(x)),magic[4:7]);assert lg in range(6) and packed in (0,1,2) and ew in (2,3,4)
        group=(1<<lg) if lg else wp
        return 2064+4*((n+group-1)//group)+((n-((n+group-1)//group if packed==2 else 0))*(12 if packed else 16)+7)//8+n*ew if group>2 else 2064+n*(4+ew)
    dw,ew=int(chr(magic[4])),int(chr(magic[5]))
    return 2064+n*(2+ew)+(n//wp)*4 if magic[:4]==b'NTHR' and wp>2 else 2064+n*(dw+ew+1)


def decode_hier(data,n):
    import numpy as np
    flags,wp,ew,version=struct.unpack_from('<4I',data)
    assert flags==0 and wp>2 and n%wp==0 and ew in (2,3,4) and version==2
    ctas=n//wp;assert len(data)==2064+ctas*4+n*(2+ew)
    a=np.frombuffer(data,offset=2064,count=ctas*4,dtype='u1').reshape(ctas,4)
    bits=a[:,0].astype('int64')|(a[:,1].astype('int64')<<8)|(a[:,2].astype('int64')<<16)
    delta=(bits^(1<<23))-(1<<23);sm=a[:,3]
    bases=np.frombuffer(data,offset=16,count=256,dtype='<u8');starts=bases[sm]+delta.astype('uint64')
    local=np.frombuffer(data,offset=2064+ctas*4,count=n,dtype='<i2')
    e=np.frombuffer(data,offset=2064+ctas*4+n*2,count=n*ew,dtype='u1').reshape(n,ew);elapsed=np.zeros(n,dtype='uint32')
    for k in range(ew):elapsed|=e[:,k].astype('uint32')<<(8*k)
    out=np.empty(n,dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')]))
    out['start']=np.repeat(starts,wp)+local.astype('int64').astype('uint64');out['elapsed']=elapsed;out['sm']=np.repeat(sm,wp)
    return out.tobytes()


def decode_reference(data,n):
    import numpy as np
    flags,group,ew,version=struct.unpack_from('<4I',data)
    assert flags==0 and group>2 and ew in (2,3,4) and version in (3,4,5)
    groups=(n+group-1)//group;m=n-(groups if version==5 else 0);local_bytes=(m*(12 if version in (4,5) else 16)+7)//8
    assert len(data)==2064+4*groups+local_bytes+n*ew
    anchors=np.frombuffer(data,offset=2064,count=4*groups,dtype='u1').reshape(-1,4)
    bits=anchors[:,0].astype('int64')|(anchors[:,1].astype('int64')<<8)|(anchors[:,2].astype('int64')<<16)
    dt=(bits^(1<<23))-(1<<23);sm=anchors[:,3];bases=np.frombuffer(data,offset=16,count=256,dtype='<u8');base=bases[sm]+dt.astype('uint64')
    if version in (4,5):
        b=np.frombuffer(data,offset=2064+4*groups,count=local_bytes,dtype='u1');i=np.arange(m)*12;byte=i//8;shift=i%8
        v=(b[byte].astype('uint32')|(b[byte+1].astype('uint32')<<8))>>shift
        local=(((v&4095).astype('int64')^2048)-2048)
    else:local=np.frombuffer(data,offset=2064+4*groups,count=n,dtype='<i2').astype('int64')
    if version==5:
        expanded=np.zeros(n,dtype='int64');expanded[np.arange(n)%group!=0]=local;local=expanded
    e=np.frombuffer(data,offset=2064+4*groups+local_bytes,count=n*ew,dtype='u1').reshape(n,ew);elapsed=np.zeros(n,dtype='uint32')
    for k in range(ew):elapsed|=e[:,k].astype('uint32')<<(8*k)
    out=np.empty(n,dtype=np.dtype([('start','<u8'),('elapsed','<u4'),('sm','<u4')]))
    out['start']=np.repeat(base,group)[:n]+local.astype('uint64');out['elapsed']=elapsed;out['sm']=np.repeat(sm,group)[:n]
    return out.tobytes()


_nv_decoder=None
def decode_nvcomp(data,n,soa):
    import os
    global _nv_decoder
    if _nv_decoder is None:
        from pathlib import Path
        lib=C.CDLL(os.environ.get('NEUTRINO_NVPIPE_LIBRARY',str(Path(__file__).with_name('nvcomp_pipeline.so'))))
        _nv_decoder=lib.nvpipe_decode;_nv_decoder.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_uint,C.c_uint];_nv_decoder.restype=C.c_int
    dest=C.create_string_buffer(n*16)
    assert _nv_decoder(data,len(data),dest,n*16,2 if soa else 1,65536)==0
    return dest.raw
