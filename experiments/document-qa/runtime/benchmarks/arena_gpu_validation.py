"""Stateful GPU oracle: shape/template invalidation and producer stream changes."""
import ctypes as C,hashlib,json,os
from pathlib import Path
import torch,triton
import triton.language as tl

@triton.jit
def neutrino_arena_add(y,N:tl.constexpr):
    i=tl.program_id(0)*128+tl.arange(0,128)
    v=tl.load(y+i,i<N,0)
    tl.store(y+i,v+1,i<N)

hook=C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER']);hook.neutrino_capture_summary.argtypes=[C.c_char_p]
out=Path(os.environ['NEUTRINO_BENCH_DIR']);streams=[torch.cuda.Stream(),torch.cuda.Stream()]
y=torch.zeros(262144,device='cuda');torch.cuda.synchronize();expected=torch.zeros(262144)
shapes=[4096,262144,4096,4096]*24
steps=[]
for token in range(12):
    assert hook.neutrino_capture_mark(token)==0
    sequence=list(shapes)
    if token in (4,5):sequence[9]=262144
    if token==8:sequence=sequence[:-3]
    for i,n in enumerate(sequence):
        which=1 if token in (6,7) and i>=40 else 0
        if token in (6,7) and i==40:torch.cuda.synchronize()
        with torch.cuda.stream(streams[which]):neutrino_arena_add[(triton.cdiv(n,128),)](y,n,num_warps=4)
        expected[:n]+=1
    torch.cuda.synchronize();steps.append(dict(token=token,launches=len(sequence)))
assert hook.neutrino_ablation_boundary()==0
actual=y.cpu();assert torch.equal(actual,expected),'Stateful kernel was lost, duplicated, or reordered unsafely'
assert hook.neutrino_capture_summary(str(out/'capture-summary.json').encode())==0
(out/'validation.json').write_text(json.dumps(dict(passed=True,steps=steps,launches=sum(s['launches'] for s in steps),output_digest=hashlib.sha256(actual.numpy().tobytes()).hexdigest()),indent=2)+'\n')
assert hook.neutrino_ablation_release()==0
