"""Independent host oracle for multidimensional CTA and partial-warp identity."""
# Reuse the checked CUDA bindings; event tests run first and retain their summary.
exec(compile((__import__('pathlib').Path(__file__).parent/'core_event_validation.py').read_text(), 'core_event_validation.py', 'exec'))
before='''
.reg .u32 %id<12>;
mov.u32 %id1, %ctaid.x;
mov.u32 %id2, %ctaid.y;
mov.u32 %id3, %ctaid.z;
mov.u32 %id4, %nctaid.x;
mov.u32 %id5, %nctaid.y;
mad.lo.u32 %id6, %id3, %id5, %id2;
mad.lo.u32 %id7, %id6, %id4, %id1;
cvt.u64.u32 %NR0, %id7;
'''
after='''
mov.u32 %id1, %tid.x;
mov.u32 %id2, %tid.y;
mov.u32 %id3, %tid.z;
mov.u32 %id4, %ntid.x;
mov.u32 %id5, %ntid.y;
mad.lo.u32 %id6, %id3, %id5, %id2;
mad.lo.u32 %id7, %id6, %id4, %id1;
cvt.u64.u32 %NR1, %id7;
mov.u64 %NR2, 123;
SAVE [T] {%NR0, %NR1, %NR2};
'''
identity=Probe('identity','warp',['kernel'],before,after)
plain='.version 8.4\n.target '+a.arch+'\n.address_size 64\n.visible .entry events(\n.param .u64 dummy\n)\n{\nret;\n}'
rows=[]
for mode in ['corrected','shift','late','vector','late_vector']:
    f=compile_ptx('ids3d-'+mode,probing(plain,[identity],maps,3,mode))
    for block in [(3,11,1),(4,4,3),(2,4,5),(4,8,2)]:
        grid=(2,3,2);threads=int(np.prod(block));ctas=int(np.prod(grid));count=ctas*((threads+31)//32);trace=mem(count*16+128);write(trace,b'\xa5'*(count*16+128));args=[trace,trace];ar=(handle*2)(*(C.addressof(x) for x in args));ok(launch(f,*grid,*block,0,None,ar,None));raw=read(trace,count*16+128)
        expected=[(b,t,123) for b in range(ctas) for t in range(0,threads,32)];actual=[struct.unpack_from('<QII',raw,i*16) for i in range(count)]
        assert actual==expected and raw[-128:]==b'\xa5'*128
        rows.append(dict(mode=mode,grid=grid,block=block,records=count,exact=True));ok(free(trace))
(a.output/'3d-summary.json').write_text(json.dumps(dict(passed=True,cases=rows),indent=2)+'\n');print(json.dumps(dict(three_d_cases=len(rows),passed=True)))
