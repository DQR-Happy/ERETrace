"""CPU-only arithmetic and output invariants of async diagnostics."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

class CaptureDetailTest(unittest.TestCase):
    def test_accounting_and_roundtrip(self):
        root=Path(__file__).resolve().parents[1]
        program=r'''
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <time.h>
        #include <assert.h>
        static void nt_json_string(FILE*f,const char*s){fprintf(f,"\"%s\"",s);}
        #include "capture_detail.h"
        int main(void){
            cd_state*d=calloc(1,sizeof(*d));cd_enabled=1;
            unsigned shape[7]={1,1,1,32,1,1,0};
            int p=cd_find(d,"kernel",shape);cd_token=1;int q=cd_find(d,"kernel",shape);assert(p!=q);
            cd_start(d);cd_stall s={.incoming=q,.slot=p,.head=q,.requested_bytes=512,.head_bytes=9723904};
            cd_stall_add(d,&s,99);cd_start(d);assert(d->nstalls==0);
            cd_stall_add(d,&s,103);cd_stall_add(d,&s,204);
            s.head_bytes=64;cd_stall_add(d,&s,77);assert(d->nstalls==2);
            uint64_t t=d->begin;
            cd_transition(d,t+10);d->busy=1;d->busy_bytes=64;cd_peak(d);
            cd_transition(d,t+40);d->busy=2;d->busy_bytes=128;cd_peak(d);
            cd_transition(d,t+60);d->busy=0;d->busy_bytes=0;
            cd_transition(d,t+100);d->end=t+100;atomic_store(&d->active,0);
            cd_add(&d->groups[q].producer[CD_MODEL],17,11);cd_add(&d->groups[q].producer[CD_MODEL],33,22);
            d->groups[q].producer_calls=2;
            assert(d->occupancy_ns[0]==50&&d->occupancy_ns[1]==30&&d->occupancy_ns[2]==20);
            assert(d->busy_byte_ns==64*30+128*20);
            cd_json(stdout,d);free(d);return 0;
        }'''
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'test.c';source.write_text(program)
            subprocess.run(['cc','-O2','-I',str(root/'neutrino/src'),str(source),'-o',str(Path(tmp)/'test')],check=True,capture_output=True)
            r=json.loads(subprocess.check_output([str(Path(tmp)/'test')]))
        self.assertEqual(sum(r['occupancy_ns']),r['window_ns'])
        self.assertEqual(r['peak_live_payload_bytes'],128)
        g=r['groups'][0]
        self.assertEqual(g['phase'],'decode')
        m=g['producer']['producer_model_launch_api']
        self.assertEqual(sum(m['log2_ns_hist']),m['count'])
        self.assertEqual((m['wall_ns'],m['thread_cpu_ns'],m['min_ns'],m['max_ns']),(50,33,17,33))
        stalls=r['condition_wait_entry_owners']
        self.assertEqual((stalls[0]['waits'],stalls[0]['wall_ns'],stalls[0]['max_ns']),(2,307,204))
        self.assertEqual((stalls[1]['waits'],stalls[1]['wall_ns']),(1,77))

if __name__=='__main__':unittest.main()
