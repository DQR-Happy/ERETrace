"""Known CTA-tail labels; checks useful diagnosis without native timestamp truth."""
import ctypes as C,hashlib,json,os,time
from pathlib import Path
import torch,triton
import triton.language as tl
@triton.jit
def neutrino_controlled_tail(y, SHORT:tl.constexpr,LONG:tl.constexpr):
 p=tl.program_id(0);i=p*128+tl.arange(0,128)
 start=tl.inline_asm_elementwise('mov.u64 $0, %clock64;',constraints='=l',args=[],dtype=tl.uint64,is_pure=False,pack=1)
 now=start;target=tl.where(p%8==0,LONG,SHORT)
 while now-start<target:
  now=tl.inline_asm_elementwise('mov.u64 $0, %clock64;',constraints='=l',args=[],dtype=tl.uint64,is_pure=False,pack=1)
 v=tl.load(y+i);tl.store(y+i,v+1)
out=Path(os.environ['NEUTRINO_BENCH_DIR']);y=torch.zeros(4096*128,device='cuda');torch.cuda.synchronize();hook=None
if os.environ.get('NEUTRINO_HOOK_DRIVER'):
 hook=C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER']);hook.neutrino_capture_summary.argtypes=[C.c_char_p]
rows=[]
for i in range(7):
 begin=time.perf_counter_ns();neutrino_controlled_tail[(4096,)](y,512,32768,num_warps=4);torch.cuda.synchronize()
 if hook:assert hook.neutrino_ablation_boundary()==0
 rows.append(time.perf_counter_ns()-begin)
a=y.cpu();assert torch.equal(a,torch.full_like(a,7))
if hook:assert hook.neutrino_capture_summary(str(out/'capture-summary.json').encode())==0
(out/'workload.json').write_text(json.dumps(dict(passed=True,cta_count=4096,warps_per_cta=4,tail_modulus=8,short_cycles=512,long_cycles=32768,launches=7,warmup=2,measured_ns=rows[2:],digest=hashlib.sha256(a.numpy().tobytes()).hexdigest()),indent=2)+'\n')
if hook:assert hook.neutrino_ablation_release()==0
