"""In-process deterministic Qwen prefill/decode; no HTTP or XTrace dependency.

Use with run_timing_matrix.py. Adapter source is supplied in this directory.
Application synchronization and greedy token selection are identical in all modes.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import resource

parser = argparse.ArgumentParser()
parser.add_argument('--adapter', required=True)
parser.add_argument('--backend', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--requests', type=int, default=3)
parser.add_argument('--warmup', type=int, default=2)
parser.add_argument('--output-tokens', type=int, default=32)
args = parser.parse_args()
if args.requests < 1 or args.warmup < 0 or args.output_tokens < 1:
    parser.error('invalid request/token count')
os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'
out = Path(os.environ['NEUTRINO_BENCH_DIR'])
lib = C.CDLL(args.adapter)
for name, types, result in [
    ('qwen_create', [C.c_char_p, C.c_char_p], C.c_void_p),
    ('qwen_prefill', [C.c_void_p, C.c_char_p], C.c_int),
    ('qwen_argmax', [C.c_void_p], C.c_int),
    ('qwen_step', [C.c_void_p, C.c_int, C.c_int], C.c_int),
    ('qwen_destroy', [C.c_void_p], None),
]:
    func = getattr(lib, name)
    func.argtypes, func.restype = types, result

# Fixed semantic prompt, with the ACTUAL tokenized length recorded below.
# This is not an AIPerf ISL=128 experiment.
prompt = (b'<|im_start|>user\nExplain how a GPU executes many threads and why '
          b'memory bandwidth matters for neural network inference. Give a concise '
          b'answer with a practical example.<|im_end|>\n<|im_start|>assistant\n'
          b'<think>\n\n</think>\n\n')
handle = lib.qwen_create(args.model.encode(), args.backend.encode())
assert handle, 'Qwen model could not be loaded on GPU'

timer = None
experiment = None
profiler = None
if os.environ.get('NEUTRINO_PROFILE') == '1':
    profiler = C.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08')
if int(os.environ.get('NEUTRINO_ABLATION_MODE', '0')):
    experiment = C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'])
    experiment.neutrino_ablation_flush.argtypes = [C.c_char_p]
if os.environ.get('NEUTRINO_TIMING') == '1':
    timer = C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'])
    timer.neutrino_timing_flush.restype = C.c_int
    timer.neutrino_timing_reset.restype = C.c_int


def snapshot(name):
    if timer:
        assert timer.neutrino_timing_flush() == 0
        source = Path(os.environ['NEUTRINO_TIMING_DIR']) / f'timing-{os.getpid()}.json'
        shutil.copyfile(source, out / name)


def request():
    begin = time.perf_counter_ns()
    count = lib.qwen_prefill(handle, prompt)
    assert count > 0, count
    tokens = [lib.qwen_argmax(handle)]
    first = time.perf_counter_ns()
    for i in range(args.output_tokens - 1):
        rc = lib.qwen_step(handle, tokens[-1], count + i)
        assert rc == 0, rc
        tokens.append(lib.qwen_argmax(handle))
    if experiment:
        assert experiment.neutrino_ablation_boundary() == 0
    end = time.perf_counter_ns()
    return dict(input_tokens=count, output_tokens=len(tokens), tokens=tokens,
                latency_ns=end - begin, prefill_first_token_ns=first - begin,
                decode_ns=end - first)


try:
    warmup = [request() for _ in range(args.warmup)]
    snapshot('startup-timing.json')
    if experiment:
        if os.environ.get('NEUTRINO_ABLATION_STATS') == '1':
            assert experiment.neutrino_ablation_flush(str(out/'startup-ablation.json').encode()) == 0
        assert experiment.neutrino_ablation_reset() == 0
    if timer:
        assert timer.neutrino_timing_reset() == 0
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    if profiler:
        assert profiler.cuProfilerStart() == 0
    begin = time.perf_counter_ns()
    measured = [request() for _ in range(args.requests)]
    wall_ns = time.perf_counter_ns() - begin
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    if profiler:
        assert profiler.cuProfilerStop() == 0
    if experiment and os.environ.get('NEUTRINO_ABLATION_STATS') == '1':
        assert experiment.neutrino_ablation_flush(str(out/'measurement-ablation.json').encode()) == 0
    snapshot('measurement-timing.json')
    if timer:
        assert timer.neutrino_timing_reset() == 0
    digest = hashlib.sha256(json.dumps([r['tokens'] for r in measured]).encode()).hexdigest()
    for row in measured:
        row.pop('tokens')
    payload = dict(schema_version=1, workload='qwen_in_process', wall_ns=wall_ns,
                   requests=measured, output_digest=digest, warmup_requests=len(warmup),
                   output_token_policy='fixed length, greedy; EOS does not stop this benchmark',
                   cuda_graphs=False, model_path=args.model,
                   prompt_sha256=hashlib.sha256(prompt).hexdigest())
    payload['resource_delta'] = {key: getattr(usage_after,key)-getattr(usage_before,key)
        for key in ('ru_utime','ru_stime','ru_minflt','ru_majflt','ru_nvcsw','ru_nivcsw')}
    (out / 'workload.json').write_text(json.dumps(payload, indent=2) + '\n')
finally:
    if experiment:
        assert experiment.neutrino_ablation_release() == 0
    lib.qwen_destroy(handle)
