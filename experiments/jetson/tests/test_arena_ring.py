"""Exercise the actual byte-ring allocator against live interval invariants."""
from pathlib import Path
import subprocess,tempfile,unittest

class ArenaRingTest(unittest.TestCase):
    def test_wrap_capacity_and_fifo_release(self):
        text=(Path(__file__).resolve().parents[1]/'neutrino/src/arena_capture.h').read_text()
        # Extract only alignment and allocator code, even when unrelated helpers
        # are inserted between ar_fit and the following accounting function.
        end=text.index('\nstatic ',text.index('static int ar_fit')+1)
        code=text[text.index('static size_t ar_align'):end]
        source=r'''
        #include <stdint.h>
        #include <stddef.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <assert.h>
        typedef struct{uint64_t tail,head;size_t capacity;}ar_ring;
        static void async_fatal(const char*s){fprintf(stderr,"%s",s);abort();}
        '''+code+r'''
        int main(void){
          ar_ring r={.capacity=4096};uint64_t b[1024],e[1024];unsigned head=0,tail=0;
          uint32_t rng=21042;
          for(unsigned step=0;step<200000;step++){
            rng=rng*1664525u+1013904223u;size_t n=(rng%256+1)*16;if((rng&7)==0)n=0;
            uint64_t start,end;
            while(!ar_fit(&r,n,&start,&end)){
              assert(head<tail);if(r.head<e[head%1024])r.head=e[head%1024];head++;
            }
            assert(end-start==n&&end-r.head<=r.capacity);
            size_t low=start%r.capacity,high=low+n;assert(high<=r.capacity);
            for(unsigned i=head;i<tail;i++){
              size_t l=b[i%1024]%r.capacity,h=l+e[i%1024]-b[i%1024];if(high>low&&h>l)assert(high<=l||h<=low);
            }
            b[tail%1024]=start;e[tail%1024]=end;tail++;r.tail=end;
            if((rng&3)==0&&head<tail){if(r.head<e[head%1024])r.head=e[head%1024];head++;}
          }
          while(head<tail){if(r.head<e[head%1024])r.head=e[head%1024];head++;}assert(r.head==r.tail);
          puts("200000 allocations, no overlap or capacity escape");return 0;
        }'''
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'test.c').write_text(source)
            subprocess.run(['cc','-O2',str(p/'test.c'),'-o',str(p/'test')],check=True)
            self.assertIn('200000 allocations',subprocess.check_output([str(p/'test')],text=True))

if __name__=='__main__':unittest.main()
