"""Independent byte-exact oracle for multi-SM clocks, overflow and raw fallback."""
import ctypes as C
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


class Codec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();p=Path(cls.tmp.name)
        header=Path(__file__).resolve().parents[1]/'neutrino/src/capture_codec.h'
        (p/'bridge.c').write_text(f'#include "{header}"\nsize_t pack(void*d,const void*s,size_t n){{return nt_pack_records(d,s,n,0,0);}}\nint unpack(void*d,size_t n,const void*s,size_t p){{return nt_unpack_records(d,n,s,p);}}\n')
        subprocess.run(['cc','-O3','-shared','-fPIC',str(p/'bridge.c'),'-o',str(p/'codec.so')],check=True)
        cls.lib=C.CDLL(str(p/'codec.so'))
        cls.lib.pack.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];cls.lib.pack.restype=C.c_size_t
        cls.lib.unpack.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t];cls.lib.unpack.restype=C.c_int

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def check_records(self,rows,packed=True):
        raw=b''.join(struct.pack('<QII',*r) for r in rows);buf=C.create_string_buffer(len(raw));n=self.lib.pack(buf,raw,len(raw))
        if not packed:self.assertEqual(n,0);return
        self.assertGreater(n,0);self.assertLess(n,len(raw));out=C.create_string_buffer(len(raw))
        self.assertEqual(self.lib.unpack(out,len(raw),buf,n),0);self.assertEqual(out.raw,raw)
        data=buf.raw[:n];count,=struct.unpack('<H',data[:2]);pos=2;bases={}
        for _ in range(count):sm=data[pos];bases[sm]=struct.unpack('<Q',data[pos+1:pos+9])[0];pos+=9
        decoded=[]
        for delta,elapsed,sm in struct.iter_unpack('<IIB',data[pos:]):decoded.append((0,0,0) if sm==255 else (bases[sm]+delta,elapsed,sm))
        self.assertEqual(decoded,rows)

    def test_unsynchronized_sm_epochs(self):self.check_records([(1_000_000+i,i,0) if i%2 else (2**55+i,i,81) for i in range(100)])
    def test_zero_records(self):self.check_records([(0,0,0)]*100)
    def test_zero_and_sm0(self):self.check_records([(0,0,0),(2**48,0,0)]*100)
    def test_uint64_endpoint(self):self.check_records([(2**64-1,2**32-1,254)]*100)
    def test_range_fallback(self):self.check_records([(1,1,0),(2**32+2,1,0)]*100,False)
    def test_smid_fallback(self):self.check_records([(1,1,255)]*100,False)
    def test_small_expansion_fallback(self):self.check_records([(1,1,1)],False)
    def test_random_roundtrip(self):
        r=random.Random(21042)
        for _ in range(100):
            bases={sm:r.randrange(2**40,2**60) for sm in range(82)}
            rows=[]
            for _ in range(1000):
                sm=r.randrange(82);rows.append((bases[sm]+r.randrange(2**30),r.randrange(2**32),sm))
            self.check_records(rows)
    def test_reject_corrupt_dictionary(self):
        data=struct.pack('<H',2)+struct.pack('<BQ',1,10)+struct.pack('<BQ',1,20)+struct.pack('<IIB',0,1,1)
        self.assertEqual(self.lib.unpack(C.create_string_buffer(16),16,data,len(data)),-1)


if __name__=='__main__':unittest.main()
