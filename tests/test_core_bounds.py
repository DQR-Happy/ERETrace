import copy,unittest
from neutrino.probe.core_bounds import Program,specialize,check_certificate,Unsupported
from neutrino.probe.core_certified import compile_layout

PTX='''.version 8.4
.target sm_86
.address_size 64
.visible .entry f(.param .u64 data,.param .u32 limit)
{
.reg .u32 %i,%j,%v,%lim;
.reg .u64 %p;
.reg .pred %q;
ld.param.u64 %p, [data];
ld.param.u32 %lim, [limit];
mov.u32 %i, %tid.x;
setp.ge.u32 %q, %i, %lim;
@%q ret;
mov.u32 %j, 0;
L:
ld.global.u32 %v, [%p];
add.u32 %j, %j, 1;
setp.lt.u32 %q, %j, %lim;
@%q bra L;
st.global.u32 [%p], %v;
ret;
}'''
class BoundsTests(unittest.TestCase):
 def test_derived_bound_and_partition(self):
  for n in [1,3,5]:
   d=specialize(PTX,{'limit':n},(2,1,1),(5,1,1))
   self.assertEqual(d['capacities'],([n+1]*n+[0]*(5-n))*2)
   for m in ['uniform','table','classes']:
    code,lay=compile_layout(PTX,d,{'limit':n},(2,1,1),(5,1,1),m)
    self.assertEqual(sum(lay['caps']),lay['slots'])
    self.assertEqual([off+j for off,c in zip(lay['offsets'],lay['caps']) for j in range(c)],list(range(lay['slots'])))
 def test_binding_and_tamper(self):
  d=specialize(PTX,{'limit':3},(1,1,1),(5,1,1))
  for src,params,grid in [(PTX+'\n',{'limit':3},(1,1,1)),(PTX,{'limit':4},(1,1,1)),(PTX,{'limit':3},(2,1,1))]:
   with self.assertRaises(Unsupported):check_certificate(src,d,params,grid,(5,1,1))
  d['capacities'][0]-=1
  with self.assertRaises(Unsupported):check_certificate(PTX,d,{'limit':3},(1,1,1),(5,1,1))
 def test_unknown_bound_not_silently_truncated(self):
  with self.assertRaises(Unsupported):Program(PTX).walk({},(1,1,1),(1,1,1),(0,0,0),(0,0,0),budget=250)
  with self.assertRaises(Unsupported):specialize(PTX.replace('add.u32 %j, %j, 1;','add.u32 %j, %j, 0;'),{'limit':3},(1,1,1),(1,1,1))
 def test_finite_data_bound(self):
  src=PTX.replace('ld.param.u32 %lim, [limit];','ld.global.u32 %lim, [%p];\nand.b32 %lim, %lim, 3;\nadd.u32 %lim, %lim, 1;')
  d=specialize(src,{},(1,1,1),(4,1,1))
  self.assertEqual(d['capacities'],[6]*4)
 def test_bad_semantics_rejected(self):
  for src in [PTX.replace('add.u32','add.sat.u32'),PTX.replace('ld.global.u32 %v, [%p];','call f;'),PTX.replace('bra L','brx.idx %i, L'),PTX.replace('mov.u32 %j, 0;','bar.red.popc.u32 %j, 0, %q;'),PTX.replace('ld.global.u32','ld.volatile.global.u32')]:
   with self.assertRaises(Unsupported):Program(src)
  with self.assertRaises(Unsupported):specialize(PTX,{'limit':3},(2**30,1,1),(4,1,1))
 def test_signed_and_wrap(self):
  src=PTX.replace('mov.u32 %j, 0;','mov.u32 %j, 4294967295;').replace('ld.global.u32 %v, [%p];','add.u32 %j, %j, 1;\nld.global.u32 %v, [%p];').replace('add.u32 %j, %j, 1;\nsetp','setp')
  self.assertEqual(specialize(src,{'limit':2},(1,1,1),(1,1,1))['capacities'],[4])
 def test_mask_bound_covers_exhaustive_concrete_inputs(self):
  # Independent concrete loop count, including threads taking the early exit.
  for mask in range(8):
   src=PTX.replace('ld.param.u32 %lim, [limit];',f'ld.global.u32 %lim, [%p];\nand.b32 %lim, %lim, {mask};\nadd.u32 %lim, %lim, 1;')
   cert=specialize(src,{},(1,1,1),(9,1,1))
   for data in range(256):
    limit=(data&mask)+1
    for tid,cap in enumerate(cert['capacities']):
     actual=1 if tid>=limit else 1+limit+1
     self.assertLessEqual(actual,cap)
 def test_three_dimensional_control_cache(self):
  # A cache keyed only by x would under-allocate CTA y=1.
  src=PTX.replace('ld.param.u32 %lim, [limit];','mov.u32 %lim, %ctaid.y;\nadd.u32 %lim, %lim, 1;')
  c=specialize(src,{},(2,2,1),(3,1,1))
  self.assertEqual(c['capacities'],[2,0,0,2,0,0,3,3,0,3,3,0])
 def test_scoped_register_aliases_rejected(self):
  src=PTX.replace('mov.u32 %j, 0;','{\n.reg .u32 %lim;\nmov.u32 %lim, 1;\n}\nmov.u32 %j, 0;')
  for bad in [src,PTX.rstrip()[:-1],PTX+'\nret;']:
   with self.assertRaises(Unsupported):Program(bad)
if __name__=='__main__':unittest.main()
