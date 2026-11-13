"""Small, idempotent GPU workload for fixed/dynamic/fallback/callback validation."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
import triton
import triton.language as tl

parser = argparse.ArgumentParser()
parser.add_argument('--loops', type=int, default=1000)
parser.add_argument('--warmup', type=int, default=8)
args = parser.parse_args()
assert args.loops > 0 and args.warmup >= 0
out = Path(os.environ['NEUTRINO_BENCH_DIR'])


@triton.jit
def neutrino_timing_vector(x, y, N: tl.constexpr):
    offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
    values = tl.load(x + offsets, offsets < N, 0)
    tl.store(y + offsets, values + 1, offsets < N)


x = torch.arange(4096, dtype=torch.float32).cuda()
y = torch.empty_like(x)


def launch():
    neutrino_timing_vector[(32,)](x, y, 4096, num_warps=4)


for _ in range(args.warmup):
    launch()
torch.cuda.synchronize()
timer = None
if os.environ.get('NEUTRINO_TIMING') == '1':
    timer = C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'])


def snapshot(name):
    if timer:
        assert timer.neutrino_timing_flush() == 0
        source = Path(os.environ['NEUTRINO_TIMING_DIR']) / f'timing-{os.getpid()}.json'
        shutil.copyfile(source, out / name)


snapshot('startup-timing.json')
if timer:
    assert timer.neutrino_timing_reset() == 0
begin = time.perf_counter_ns()
for _ in range(args.loops):
    launch()
torch.cuda.synchronize()
wall_ns = time.perf_counter_ns() - begin
snapshot('measurement-timing.json')
if timer:
    assert timer.neutrino_timing_reset() == 0
result = y.cpu()
assert torch.equal(result, torch.arange(4096, dtype=torch.float32) + 1)
payload = dict(schema_version=1, workload='triton_vector', wall_ns=wall_ns,
               loops=args.loops, warmup=args.warmup,
               output_digest=hashlib.sha256(result.numpy().tobytes()).hexdigest())
(out / 'workload.json').write_text(json.dumps(payload, indent=2) + '\n')
