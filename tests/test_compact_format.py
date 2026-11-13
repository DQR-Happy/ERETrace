import struct,unittest
from test_reference_format import codec,checks
class CompactFormat(unittest.TestCase):
 def test_zero_elision_and_partial_group(self):
  for n in [1,2,3,4,5,7,8,9,31,33,129]:
   for k in [3,4,5,7,8,32]:
    g=(n+k-1)//k;m=n-g;nb=(m*12+7)//8;p=bytearray(2064+4*g+nb+2*n);struct.pack_into('<4I',p,0,0,k,2,5)
    struct.pack_into('<Q',p,16,2**64-100);bits=q=0;raw=[]
    for i in range(n):
     group=i//k;local=0 if i%k==0 else [-2048,2047,-1,1][i%4]
     if i%k==0:p[2064+4*group:2067+4*group]=(group*7).to_bytes(3,'little')
     else:bits|=(local&4095)<<(12*q);q+=1
     struct.pack_into('<H',p,2064+4*g+nb+2*i,65535-i)
     raw.append(struct.pack('<QII',(2**64-100+group*7+local)%2**64,65535-i,0))
    p[2064+4*g:2064+4*g+nb]=bits.to_bytes(nb,'little');self.assertEqual(codec.decode_flex(bytes(p),n),b''.join(raw))
    self.assertEqual(len(p),checks.format_size(b'NTRF022\n',n,k))
 def test_declared_flags_rejected(self):
  p=bytearray(2064+4+2+6);struct.pack_into('<4I',p,0,16,4,2,5)
  with self.assertRaises(AssertionError):codec.decode_flex(p,3)
if __name__=='__main__':unittest.main()
