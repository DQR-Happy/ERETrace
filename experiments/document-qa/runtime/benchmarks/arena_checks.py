"""Trace structure/sequence/byte accounting for the mixed arena experiment."""
import hashlib
import json
import struct

def sequence(path):
    result = dict(frames=0, payload=0, encoded_payload=0, nominal_payload=0, encoded_frames=0)
    signature=hashlib.sha256(); offsets=[]
    with path.open('rb') as f:
        magic=f.read(8);assert (magic in (b'NTGPU13\n',b'NTGPUV2\n',b'NTNVCA1\n',b'NTNVCS1\n') or ((magic[:4] in (b'NTFX',b'NTHR',b'NTRF') and magic[6:]==b'!\n') or (magic[:4]==b'NTRF' and magic[7:]==b'\n')))
        while True:
            start=f.tell(); h=f.read(48)
            if not h: break
            assert len(h)==48
            gx,gy,gz,bx,by,bz,shared,maps,item,div,offset=struct.unpack('<10IQ',h)
            assert min(gx,gy,gz,bx,by,bz)>0 and (maps,item,div,offset)==(1,16,32,48)
            raw=gx*gy*gz*((bx*by*bz+31)//32)*16
            packed,=struct.unpack('<Q',f.read(8));tag=packed>>62;size=packed&((1<<62)-1)
            assert (magic[:4]==b'NTNV' and tag==3 and 16<size<raw*3) or (tag==0 and size in (0,raw//16*13)) or (magic==b'NTGPUV2\n' and ((tag==2 and size==raw//16*9+2064) or (tag==1 and size==raw))) or (magic[:4] in (b'NTFX',b'NTHR',b'NTRF') and ((tag==3 and size==format_size(magic,raw//16,(bx*by*bz+31)//32)) or (tag==1 and size==raw)))
            copied=size or raw;f.seek(copied,1);assert f.tell()<=path.stat().st_size
            result['nominal_payload']+=(format_size(magic,raw//16,(bx*by*bz+31)//32) if magic[:4] in (b'NTFX',b'NTHR',b'NTRF') else raw//16*9+2064) if tag==1 else copied
            result['frames']+=1;result['payload']+=raw;result['encoded_payload']+=copied;result['encoded_frames']+=bool(packed)
            signature.update(h);offsets.append((start,raw,copied))
    result['shape_sequence_sha256']=signature.hexdigest()
    ids=path.with_suffix('.ids')
    if ids.exists():
        lines=[tuple(map(int,l.split(','))) for l in ids.read_text().splitlines()]
        assert len(lines)==len(offsets)
        assert all(lines[i][0]+1==lines[i+1][0] for i in range(len(lines)-1))
        assert all(tuple(l[2:])==off for l,off in zip(lines,offsets))
        assert lines[0][1]==0 and all(a[1]<=b[1] for a,b in zip(lines,lines[1:]))
        result['launch_ids_checked']=len(lines)
        result['token_frames']={str(t):sum(l[1]==t for l in lines) for t in sorted({l[1] for l in lines})}
    return result

def check_run(run, trace, paths, checks, workload, capture):
    if not capture:
        assert not paths
        return dict(native=True)
    assert len(paths)==workload['warmup_requests']+len(workload['requests'])-workload.get('warmup_trace_files_removed',0)
    assert all(p.suffix=='.nts' for p in paths)
    paths=sorted(paths,key=lambda p:int(p.stem.split('-')[-1]))
    seq=[sequence(p) for p in paths]
    assert len({json.dumps({k:v for k,v in s.items() if k not in ('launch_ids_checked','token_frames','encoded_payload','nominal_payload')},sort_keys=True) for s in seq})==1
    assert all(c['zero_records']==0 for c in checks)
    if any(c['record_validation'] for c in checks):
        assert all(c['record_validation'] and c['nonzero_records']>0 and set(c['observed_sm_ids'])<=set(range(82)) for c in checks)
    summary=json.loads((run/'capture-summary.json').read_text())
    n=len(workload['requests']);one=seq[0]
    if summary.get('arena_engines'):
        assert len(summary['arena_engines'])==1
        a=summary['arena_engines'][0]
        assert a['calls']==one['frames']*n and a['payload']==one['payload']*n
        assert a['copy_payload']==sum(s['nominal_payload'] for s in seq[-n:])
        assert a['storage_bytes']==sum(p.stat().st_size for p in paths[-n:])
        assert a['published_total']==a['consumed_total']==a['retired_total']
        assert a['outstanding_bytes']==0
        assert a.get('encoder_frames',a['encoders'])==one['encoded_frames']*n
        assert a['ready_records']==a['done_records']==a['stream_waits']==a['worker_syncs']==a['batches']
        assert a['transfer_bytes']>=a['copy_payload']
    return dict(per_request=one, files=len(paths), structures_checked=sum(s['frames'] for s in seq),
                sequence_equal=True,launch_ids_checked=sum(s.get('launch_ids_checked',0) for s in seq))


def format_size(magic,n,wp):
    if magic[:4]==b'NTRF':
        lg,packed,ew=map(lambda x:int(chr(x)),magic[4:7]);assert lg in range(6) and packed in (0,1,2) and ew in (2,3,4)
        group=(1<<lg) if lg else wp
        return 2064+4*((n+group-1)//group)+((n-((n+group-1)//group if packed==2 else 0))*(12 if packed else 16)+7)//8+n*ew if group>2 else 2064+n*(4+ew)
    dw,ew=int(chr(magic[4])),int(chr(magic[5]))
    return 2064+n*(2+ew)+(n//wp)*4 if magic[:4]==b'NTHR' and wp>2 else 2064+n*(dw+ew+1)
