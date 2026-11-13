"""Public file recovery at uint64 boundaries and explicit raw fallback tags."""
import importlib.util,struct,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    s=importlib.util.spec_from_file_location(name,ROOT/'benchmarks'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
codec=load('trace_codec');checks=load('arena_checks')
class GPU9Format(unittest.TestCase):
    def test_roundtrip_and_raw_fallback(self):
        # 33 threads -> two warp slots; mixed sign deltas around unsigned wrap.
        rows=[(2**64-1,2**32-1,255),(3,7,255)]
        p=bytearray(2064+18);struct.pack_into('<Q',p,16+255*8,1)
        struct.pack_into('<ii',p,2064,-2,2);struct.pack_into('<II',p,2072,2**32-1,7);p[2080:]=bytes([255,255])
        raw=b''.join(struct.pack('<QII',*r) for r in rows)
        self.assertEqual(codec.decode_gpu9(bytes(p),2),raw)
        h=struct.pack('<10IQ',1,1,1,33,1,1,0,1,16,32,48)
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'arena-0.nts';f.write_bytes(b'NTGPUV2\n'+h+struct.pack('<Q',(1<<63)|len(p))+p+h+struct.pack('<Q',(1<<62)|len(raw))+raw)
            a=checks.sequence(f);b=codec.check_gpu(f,True)
            self.assertEqual(a['frames'],2);self.assertEqual(a['payload'],64)
            self.assertEqual(a['nominal_payload'],2*len(p));self.assertEqual(b['nonzero_records'],4)
            self.assertEqual(b['observed_sm_ids'],[255])
    def test_invalid_size_rejected(self):
        h=struct.pack('<10IQ',1,1,1,32,1,1,0,1,16,32,48)
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'arena-0.nts';f.write_bytes(b'NTGPUV2\n'+h+struct.pack('<Q',(1<<63)|100)+b'x'*100)
            with self.assertRaises(AssertionError):checks.sequence(f)
    def test_error_flag_rejected(self):
        p=bytearray(2073);struct.pack_into('<I',p,0,1)
        with self.assertRaises(AssertionError):codec.decode_gpu9(bytes(p),1)
if __name__=='__main__':unittest.main()
