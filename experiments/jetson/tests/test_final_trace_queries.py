import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from final_trace_queries import analyze_files,queries

class FinalQueryTests(unittest.TestCase):
    def setUp(self):
        self.rows=[(7,0,0),(8,1,0),(9,2,81),(10,3,81),(11,65535,255),(12,2**32-1,255)]
        self.raw=b''.join(struct.pack('<QII',*r) for r in self.rows)
        self.hist=[0]*33
        for b in [0,1,2,2,16,32]:self.hist[b]+=1
        self.expected=dict(records=6,cycle_sum=4295032836,cycle_max=2**32-1,
            log2_hist=self.hist,per_sm={'0':[2,1],'81':[2,5],'255':[2,4295032830]})
    def test_integer_extremes(self):
        self.assertEqual(queries(self.raw),self.expected)
    def test_container_and_bad_oracle(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'arena-0.nts'
            h=struct.pack('<10IQ',6,1,1,32,1,1,0,1,16,32,48)
            p.write_bytes(b'NTGPU13\n'+h+struct.pack('<Q',0)+self.raw)
            p.with_suffix('.ids').write_text('0,7,8,96,96\n')
            line='0,7,6,4295032836,4294967295;'+','.join(map(str,self.hist))+';0:2:1,81:2:5,255:2:4295032830,\n'
            p.with_suffix('.rawq').write_text(line)
            a=analyze_files([p]);self.assertTrue(a['passed']);self.assertEqual(a['total_records'],6)
            self.assertEqual(a['files'][0]['tokens']['7']['cycle_max'],2**32-1)
            p.with_suffix('.rawq').write_text(line.replace('4295032836','4295032837'))
            with self.assertRaises(AssertionError):analyze_files([p])
    def test_field_layout(self):
        # Identical queries after a column-layout round trip, including >32-bit sums.
        import numpy as np
        a=np.frombuffer(self.raw,dtype=[('start','<u8'),('elapsed','<u4'),('sm','<u4')])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'arena-0.nts';data=a['start'].tobytes()+a['elapsed'].tobytes()+a['sm'].astype('u1').tobytes()
            p.write_bytes(b'NTGPU13\n'+struct.pack('<10IQ',6,1,1,32,1,1,0,1,16,32,48)+struct.pack('<Q',78)+data)
            from final_trace_queries import frames
            self.assertEqual(queries(next(frames(p))[2]),self.expected)

if __name__=='__main__':unittest.main()
