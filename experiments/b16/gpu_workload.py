#!/usr/bin/env python3
"""Fixed-token exact-length single/batched llama.cpp workload for 3090 runs."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

# The runner invokes this file by absolute path with cwd=source.  Make the
# repository's trace validator explicit instead of relying on cwd sys.path.
sys.path.insert(0, '/root/autodl-tmp/neutrino-b16-attribution-20261007/collector-old/benchmarks')

p = argparse.ArgumentParser()
p.add_argument('--adapter', required=True)
p.add_argument('--backend', required=True)
p.add_argument('--model', required=True)
p.add_argument('--requests', type=int, default=3)
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--input-tokens', type=int, required=True)
p.add_argument('--output-tokens', type=int, required=True)
p.add_argument('--batch-size', type=int, default=1)
a = p.parse_args()
if min(a.requests, a.input_tokens, a.output_tokens, a.batch_size) < 1 or a.batch_size > 16:
    p.error('invalid workload size')
os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'
out = Path(os.environ['NEUTRINO_BENCH_DIR'])
lib = C.CDLL(os.environ.get('B16_OLD_ADAPTER', a.adapter))
lib.qwen_create.argtypes = [C.c_char_p, C.c_char_p]
lib.qwen_create.restype = C.c_void_p
lib.qwen_destroy.argtypes = [C.c_void_p]
lib.qwen_fill_token_ids.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int)]
lib.qwen_fill_token_ids.restype = C.c_int
deep=C.CDLL(os.environ['DEEP_CUPTI_LIBRARY'])
deep.deep_stop.argtypes=[C.c_char_p]
assert deep.deep_init()==0
handle = lib.qwen_create(a.model.encode(), a.backend.encode())
assert handle, 'llama.cpp model/context creation failed'
lib.qwen_prefill_ids.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int]
lib.qwen_prefill_ids.restype = C.c_int
lib.qwen_argmax.argtypes = [C.c_void_p]
lib.qwen_argmax.restype = C.c_int
lib.qwen_step.argtypes = [C.c_void_p, C.c_int, C.c_int]
lib.qwen_step.restype = C.c_int
lib.qwen_batch_prefill.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_int, C.POINTER(C.c_int)]
lib.qwen_batch_prefill.restype = C.c_int
lib.qwen_batch_argmax.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_int)]
lib.qwen_batch_argmax.restype = C.c_int
lib.qwen_batch_step.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_int, C.POINTER(C.c_int)]
lib.qwen_batch_step.restype = C.c_int

stream_trace = os.environ.get('NEUTRINO_BENCH_STREAM_TRACE') == '1'
streamed_trace_requests = []
streamed_trace_files_removed = 0

def closed_trace_paths():
    root = Path(os.environ.get('NEUTRINO_TRACEDIR', out / 'no-traces'))
    if not root.exists():
        return []
    result = []
    for suffix in ('.bin', '.ntz', '.ntp', '.nts'):
        result.extend(root.glob(f'*/result/*{suffix}'))
    return sorted(result)

def validate_and_remove_closed_trace(label):
    """Validate closed per-request frames, retain only counters, then free space."""
    from run_arena_matrix import trace_check
    paths = closed_trace_paths()
    checks = [trace_check(path, False) for path in paths]
    if os.environ.get('B16_TRACE_DISTRIBUTION') == '1':
        from trace_distribution import collect
        for path in paths: collect(path, out / 'trace-distribution.jsonl', label)
    frames = sum(int(row.get('frames', 0)) for row in checks)
    byte_count = sum(int(row.get('bytes', 0)) for row in checks)
    logical_bytes = sum(int(row.get('logical_bytes', row.get('payload', 0))) for row in checks)
    for path in paths:
        path.unlink()
        path.with_suffix('.ids').unlink(missing_ok=True)
    summary = {'request': label, 'files': len(paths), 'trace_bytes': byte_count,
               'frames': frames, 'logical_bytes': logical_bytes,
               'checks': checks, 'validated_before_delete': True}
    if ablation and (not paths or frames <= 0):
        raise RuntimeError(f'capture request {label} produced no checkable trace')
    return summary

one = (C.c_int * a.input_tokens)()
assert lib.qwen_fill_token_ids(handle, a.input_tokens, one) == a.input_tokens
token_ids = (C.c_int * (a.input_tokens * a.batch_size))()
for s in range(a.batch_size):
    token_ids[s*a.input_tokens:(s+1)*a.input_tokens] = one[:]
ablation = None
if int(os.environ.get('NEUTRINO_ABLATION_MODE', '0')):
    ablation = C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'])
    ablation.neutrino_ablation_boundary.restype = C.c_int
    mark = getattr(ablation, 'neutrino_capture_mark', None)
    if mark:
        mark.argtypes = [C.c_int]
        mark.restype = C.c_int

def run_request():
    if ablation and mark and os.environ.get('B16_TRACE_DISTRIBUTION') == '1':
        assert mark(0) == 0
    t0 = time.perf_counter_ns()
    if a.batch_size == 1:
        rc = lib.qwen_prefill_ids(handle, one, a.input_tokens)
        assert rc == a.input_tokens, f'prefill rc={rc}'
        generated = [lib.qwen_argmax(handle)]
    else:
        rows = (C.c_int * a.batch_size)()
        rc = lib.qwen_batch_prefill(handle, token_ids, a.batch_size, a.input_tokens, rows)
        assert rc == 0, f'batch prefill rc={rc}'
        first = (C.c_int * a.batch_size)()
        assert lib.qwen_batch_argmax(handle, rows, a.batch_size, first) == a.batch_size
        generated = [list(first)]
    prefill_done = time.perf_counter_ns()
    for step in range(a.output_tokens - 1):
        if ablation:
            if mark:
                assert mark(step + 1) == 0
        if a.batch_size == 1:
            rc = lib.qwen_step(handle, generated[-1], a.input_tokens + step)
            assert rc == 0, f'decode rc={rc}'
            generated.append(lib.qwen_argmax(handle))
        else:
            prev = (C.c_int * a.batch_size)(*generated[-1])
            nxt = (C.c_int * a.batch_size)()
            assert lib.qwen_batch_step(handle, prev, a.batch_size,
                                       a.input_tokens + step, nxt) == a.batch_size
            generated.append(list(nxt))
    drain_begin = time.perf_counter_ns()
    if ablation:
        assert ablation.neutrino_ablation_boundary() == 0
    end = time.perf_counter_ns()
    per_sequence_tokens = (
        [generated] if a.batch_size == 1 else
        [[row[i] for row in generated] for i in range(a.batch_size)]
    )
    return {
        'input_tokens': a.input_tokens,
        'output_tokens': a.output_tokens,
        'batch_size': a.batch_size,
        'latency_ns': end - t0,
        'prefill_first_token_ns': prefill_done - t0,
        'decode_ns': drain_begin - prefill_done,
        'trace_drain_ns': end - drain_begin,
        'tokens': generated,
        'per_sequence_tokens': per_sequence_tokens,
    }

try:
    warmup_trace_files_removed = 0
    warmup_trace_requests_removed = 0
    warmup_trace_checks = []
    warmup = []
    for _ in range(a.warmup):
        warmup.append(run_request())
        if stream_trace or (ablation and os.environ.get('NEUTRINO_BENCH_DISCARD_WARMUP') == '1'):
            if stream_trace:
                warmup_trace_checks.append(validate_and_remove_closed_trace(f'warmup-{len(warmup)}'))
                streamed_trace_requests.append(warmup_trace_checks[-1])
                warmup_trace_files_removed += warmup_trace_checks[-1]['files']
                streamed_trace_files_removed += warmup_trace_checks[-1]['files']
            else:
                from arena_checks import sequence
                traces = list(Path(os.environ['NEUTRINO_TRACEDIR']).glob('*/result/*.nts'))
                assert len(traces) == 1, f'expected one closed warmup trace, got {len(traces)}'
                warmup_trace_checks.append(sequence(traces[0]))
                traces[0].unlink()
                traces[0].with_suffix('.ids').unlink(missing_ok=True)
                warmup_trace_files_removed += 1
            warmup_trace_requests_removed += 1
    if ablation:
        ablation.neutrino_ablation_reset.restype = C.c_int
        assert ablation.neutrino_ablation_reset() == 0
    if ablation and not stream_trace and os.environ.get('NEUTRINO_BENCH_DISCARD_WARMUP') == '1':
        from arena_checks import sequence
        traces = list(Path(os.environ['NEUTRINO_TRACEDIR']).glob('*/result/*.nts'))
        for trace in traces:
            sequence(trace)
            trace.unlink()
            trace.with_suffix('.ids').unlink(missing_ok=True)
    timer = None
    if os.environ.get('NEUTRINO_TIMING') == '1':
        timer = C.CDLL(os.environ['NEUTRINO_HOOK_DRIVER'])
        timer.neutrino_timing_flush.restype = C.c_int
        timer.neutrino_timing_reset.restype = C.c_int
        assert timer.neutrino_timing_flush() == 0
        src = Path(os.environ['NEUTRINO_TIMING_DIR']) / f'timing-{os.getpid()}.json'
        shutil.copyfile(src, out / 'startup-timing.json')
        if ablation:
            assert ablation.neutrino_ablation_flush(str(out/'startup-ablation.json').encode()) == 0
            assert ablation.neutrino_ablation_reset() == 0
        assert timer.neutrino_timing_reset() == 0
    assert deep.deep_start()==0
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    syscall_markers = os.environ.get('NEUTRINO_BENCH_SYSCALL_MARKERS') == '1'
    if syscall_markers:
        os.write(2, b'NEUTRINO_BENCH_WINDOW_BEGIN\n')
    t0 = time.perf_counter_ns()
    measured = []
    for request_index in range(a.requests):
        measured.append(run_request())
        if stream_trace:
            check = validate_and_remove_closed_trace(f'measured-{request_index + 1}')
            streamed_trace_requests.append(check)
            streamed_trace_files_removed += check['files']
    measurement_window_ns = time.perf_counter_ns() - t0
    if syscall_markers:
        os.write(2, b'NEUTRINO_BENCH_WINDOW_END\n')
    # Under streaming validation the harness deletes each closed request trace
    # immediately; exclude that offline validation/deletion work from request latency.
    wall = (sum(int(x['latency_ns']) for x in measured)
            if stream_trace else measurement_window_ns)
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    assert deep.deep_stop(str(out/"cupti.json").encode())==0
    if ablation:
        if os.environ.get('NEUTRINO_ABLATION_STATS') == '1':
            assert ablation.neutrino_ablation_flush(str(out/'measurement-ablation.json').encode()) == 0
        summary = getattr(ablation, 'neutrino_capture_summary', None)
        if summary:
            summary.argtypes = [C.c_char_p]
            assert summary(str(out/'capture-summary.json').encode()) == 0
    if timer:
        assert timer.neutrino_timing_flush() == 0
        src = Path(os.environ['NEUTRINO_TIMING_DIR']) / f'timing-{os.getpid()}.json'
        shutil.copyfile(src, out / 'measurement-timing.json')
        assert timer.neutrino_timing_reset() == 0
    aggregate_digest = hashlib.sha256(json.dumps([x['tokens'] for x in measured], separators=(',', ':')).encode()).hexdigest()
    per_sequence_output_digests = []
    for sequence_id in range(a.batch_size):
        payload = [row['per_sequence_tokens'][sequence_id] for row in measured]
        per_sequence_output_digests.append(hashlib.sha256(
            json.dumps(payload, separators=(',', ':')).encode()).hexdigest())
    # Use the first sequence's token digest as the matrix anchor so batch=1/4/8
    # can be compared by the shared runner; retain every sequence digest and
    # the full grouped digest separately for an explicit batch correctness check.
    digest = per_sequence_output_digests[0]
    for row in measured:
        row.pop('tokens')
        row.pop('per_sequence_tokens')
    result = {
        'schema_version': 1,
        'workload': 'llama_fixed_token_exact_length',
        'wall_ns': wall,
        'requests': measured,
        'output_digest': digest,
        'aggregate_output_digest': aggregate_digest,
        'warmup_requests': a.warmup,
        'warmup_trace_files_removed': warmup_trace_files_removed,
        'warmup_trace_requests_removed': warmup_trace_requests_removed,
        'warmup_trace_checks': warmup_trace_checks,
        'measurement_window_ns': measurement_window_ns,
        'wall_definition': 'sum_request_latencies' if stream_trace else 'measurement_window',
        'trace_files_removed_after_validation': bool(stream_trace),
        'streamed_trace_validation': {
            'enabled': bool(stream_trace), 'capture': bool(ablation),
            'requests': streamed_trace_requests,
            'files_removed': streamed_trace_files_removed,
            'trace_bytes': sum(x['trace_bytes'] for x in streamed_trace_requests),
            'frames': sum(x['frames'] for x in streamed_trace_requests),
            'logical_bytes': sum(x['logical_bytes'] for x in streamed_trace_requests),
            'all_validated_before_delete': all(x['validated_before_delete'] for x in streamed_trace_requests),
        },
        'output_token_policy': 'fixed greedy output count; EOS does not stop the benchmark',
        'cuda_graphs': False,
        'model_path': a.model,
        'input_token_ids_sha256': hashlib.sha256(bytes(str(list(one)).encode())).hexdigest(),
        'synthetic_input': True,
        'batch_size': a.batch_size,
        'per_sequence_output_digests': per_sequence_output_digests,
        'process_cpu_user_s': usage1.ru_utime - usage0.ru_utime,
        'process_cpu_system_s': usage1.ru_stime - usage0.ru_stime,
    }
    (out / 'workload.json').write_text(json.dumps(result, indent=2) + '\n')
finally:
    if ablation:
        release = getattr(ablation, 'neutrino_ablation_release', None)
        if release: assert release() == 0
    lib.qwen_destroy(handle)
