import importlib.util,struct,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'benchmarks'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
codec=load('trace_codec');checks=load('arena_checks')
class HierFormat(unittest.TestCase):
 def test_hier_wrap_and_fallback(self):
  wp=4;n=8;ew=2;p=bytearray(2064+(n//wp)*4+n*(2+ew));struct.pack_into('<4I',p,0,0,wp,ew,2);struct.pack_into('<Q',p,16+255*8,2**64-2)
  raw=[]
  for c in range(2):
   delta=(-2,7)[c];p[2064+c*4:2064+c*4+3]=(delta&0xffffff).to_bytes(3,'little');p[2064+c*4+3]=255
   for w in range(wp):
    i=c*wp+w;local=[0,-32768,32767,-1][w];struct.pack_into('<h',p,2072+2*i,local);struct.pack_into('<H',p,2088+2*i,65535-i);raw.append(struct.pack('<QII',(2**64-2+delta+local)%(2**64),65535-i,255))
  raw=b''.join(raw);self.assertEqual(codec.decode_flex(bytes(p),n),raw)
  h=struct.pack('<10IQ',2,1,1,128,1,1,0,1,16,32,48)
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'arena-0.nts';f.write_bytes(b'NTHR32!\n'+h+struct.pack('<Q',(3<<62)|len(p))+p+h+struct.pack('<Q',(1<<62)|len(raw))+raw)
   a=checks.sequence(f);b=codec.check_gpu(f,True);self.assertEqual(a['nominal_payload'],2*len(p));self.assertEqual(b['nonzero_records'],16)
 def test_flat_widths(self):
  for dw in [2,3,4]:
   for ew in [2,3,4]:
    n=3;p=bytearray(2064+n*(dw+ew+1));struct.pack_into('<4I',p,0,0,dw,ew,1);struct.pack_into('<Q',p,16+81*8,0)
    ds=[-(1<<(dw*8-1)),(1<<(dw*8-1))-1,0];raw=[]
    for i,x in enumerate(ds):
     p[2064+i*dw:2064+(i+1)*dw]=(x%(1<<(dw*8))).to_bytes(dw,'little');p[2064+n*dw+i*ew:2064+n*dw+(i+1)*ew]=((1<<(ew*8))-1).to_bytes(ew,'little');p[2064+n*(dw+ew)+i]=81;raw.append(struct.pack('<QII',x%(2**64),(1<<(ew*8))-1,81))
    self.assertEqual(codec.decode_flex(bytes(p),n),b''.join(raw))
 def test_escape_and_truncation_rejected(self):
  for p in [struct.pack('<4I',1,4,2,2)+bytes(2096),struct.pack('<4I',0,4,2,2)+bytes(10)]:
   with self.assertRaises(AssertionError):codec.decode_flex(p,8)
if __name__=='__main__':unittest.main()
