import importlib.util,struct,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'benchmarks'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
codec=load('trace_codec');checks=load('arena_checks')
class ReferenceFormat(unittest.TestCase):
 def make(self,n,g,lb,ew):
  groups=(n+g-1)//g;localbytes=(n*lb+7)//8;out=bytearray(2064+4*groups+localbytes+n*ew);struct.pack_into('<4I',out,0,0,g,ew,4 if lb==12 else 3)
  raw=[];bases={0:2**64-4,255:1234}
  for sm,b in bases.items():struct.pack_into('<Q',out,16+sm*8,b)
  packed=0
  for i in range(n):
   c=i//g;sm=255 if c%2 else 0;dt=c*10-20;local=[0,-(1<<(lb-1)),(1<<(lb-1))-1,-1][i%4];elapsed=(1<<(ew*8))-1-i%31
   if i%g==0:out[2064+4*c:2064+4*c+3]=(dt%2**24).to_bytes(3,'little');out[2064+4*c+3]=sm
   packed|=(local%(1<<lb))<<(i*lb)
   off=2064+4*groups+localbytes+i*ew;out[off:off+ew]=elapsed.to_bytes(ew,'little')
   raw.append(struct.pack('<QII',(bases[sm]+dt+local)%2**64,elapsed,sm))
  out[2064+4*groups:2064+4*groups+localbytes]=packed.to_bytes(localbytes,'little')
  return bytes(out),b''.join(raw)
 def test_partial_groups_and_exact_bit_boundaries(self):
  for n in [1,2,3,4,5,31,33,129]:
   for group in [4,8,16,32,5]:
    for lb in [12,16]:
     for ew in [2,3]:
      p,r=self.make(n,group,lb,ew);self.assertEqual(codec.decode_flex(p,n),r,(n,group,lb,ew))
 def test_container_partial_tile(self):
  n=33;p,r=self.make(n,8,12,2);header=struct.pack('<10IQ',33,1,1,32,1,1,0,1,16,32,48)
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'arena-0.nts';f.write_bytes(b'NTRF312\n'+header+struct.pack('<Q',3<<62|len(p))+p+header+struct.pack('<Q',1<<62|len(r))+r)
   x=checks.sequence(f);self.assertEqual(x['nominal_payload'],2*len(p));self.assertEqual(codec.check_gpu(f,True)['nonzero_records'],66)
 def test_bad_flags_and_lengths(self):
  p,_=self.make(33,8,12,2)
  for q in [struct.pack('<I',16)+p[4:],p[:-1],p+b'0']:
   with self.assertRaises(AssertionError):codec.decode_flex(q,33)
if __name__=='__main__':unittest.main()
