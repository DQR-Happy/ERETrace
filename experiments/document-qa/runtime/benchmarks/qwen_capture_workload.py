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
if os.environ.get('DEEP_CPU'):
    os.sched_setaffinity(0, {int(os.environ['DEEP_CPU'])})
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
deep = None
if os.environ.get('DEEP_CUPTI') == '1':
    deep = C.CDLL(str(Path(__file__).resolve().parent/'deep_cupti.so'))
    deep.deep_stop.argtypes = [C.c_char_p]
    assert deep.deep_init() == 0

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


def mark(token):
    callback = getattr(experiment, "neutrino_capture_mark", None) if experiment else None
    if callback:
        assert callback(token) == 0


def space_snapshot():
    result = {}
    for name in ('status', 'smaps_rollup'):
        values = {}
        for line in Path('/proc/self', name).read_text().splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[2] == 'kB':
                values[fields[0].rstrip(':')] = int(fields[1]) * 1024
        result[name + '_bytes'] = values
    result['threads'] = {}
    for path in Path('/proc/self/task').iterdir():
        try:
            # comm may contain spaces; stat fields after the final ')' start at 3.
            stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
            result['threads'][path.name] = dict(comm=(path/'comm').read_text().strip(),
                user_ticks=int(stat[11]), system_ticks=int(stat[12]))
        except FileNotFoundError:
            pass
    driver = experiment or C.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08')
    free, total = C.c_size_t(), C.c_size_t()
    driver.cuMemGetInfo_v2.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    assert driver.cuMemGetInfo_v2(C.byref(free), C.byref(total)) == 0
    result['cuda_free_bytes'], result['cuda_total_bytes'] = free.value, total.value
    result['clock_ticks_per_second'] = os.sysconf('SC_CLK_TCK')
    return result


def request():
    begin = time.perf_counter_ns()
    mark(0)
    count = lib.qwen_prefill(handle, prompt)
    assert count > 0, count
    tokens = [lib.qwen_argmax(handle)]
    first = time.perf_counter_ns()
    step_ns = []
    for i in range(args.output_tokens - 1):
        step_begin = time.perf_counter_ns()
        mark(i + 1)
        rc = lib.qwen_step(handle, tokens[-1], count + i)
        assert rc == 0, rc
        tokens.append(lib.qwen_argmax(handle))
        step_ns.append(time.perf_counter_ns() - step_begin)
    drain_begin = time.perf_counter_ns()
    if experiment:
        assert experiment.neutrino_ablation_boundary() == 0
    end = time.perf_counter_ns()
    return dict(input_tokens=count, output_tokens=len(tokens), tokens=tokens,
                latency_ns=end - begin, prefill_first_token_ns=first - begin,
                decode_ns=drain_begin - first, trace_drain_ns=end - drain_begin, decode_steps_ns=step_ns)


try:
    warmup = [request() for _ in range(args.warmup)]
    snapshot('startup-timing.json')
    if experiment:
        if os.environ.get('NEUTRINO_ABLATION_STATS') == '1':
            assert experiment.neutrino_ablation_flush(str(out/'startup-ablation.json').encode()) == 0
        assert experiment.neutrino_ablation_reset() == 0
    if timer:
        assert timer.neutrino_timing_reset() == 0
    space_before = space_snapshot() if os.environ.get('DEEP_SPACE') == '1' else None
    detail = experiment if os.environ.get('NEUTRINO_CAPTURE_DETAIL') == '1' else None
    if detail:
        detail.neutrino_capture_detail_stop.argtypes = [C.c_char_p]
        assert detail.neutrino_capture_detail_start() == 0
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    if profiler:
        assert profiler.cuProfilerStart() == 0
    if os.environ.get('DEEP_MARKERS') == '1':
        marker = os.open('/dev/null', os.O_WRONLY)
        os.dup2(marker, 1000)
        os.close(marker)
    if deep:
        assert deep.deep_start() == 0
    if os.environ.get('DEEP_MARKERS') == '1':
        os.write(1000, b'NDEEP_BEGIN\n')
    begin = time.perf_counter_ns()
    measured = [request() for _ in range(args.requests)]
    wall_ns = time.perf_counter_ns() - begin
    if detail:
        assert detail.neutrino_capture_detail_stop(None) == 0
    if os.environ.get('DEEP_MARKERS') == '1':
        os.write(1000, b'NDEEP_END\n')
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    space_after = space_snapshot() if space_before is not None else None
    if deep:
        assert deep.deep_stop(str(out/'cupti.json').encode()) == 0
    if profiler:
        assert profiler.cuProfilerStop() == 0
    if detail:
        assert detail.neutrino_capture_detail_stop(str(out/'capture-detail.json').encode()) == 0
    if space_before is not None:
        (out/'space.json').write_text(json.dumps(dict(before=space_before, after=space_after,
            max_rss_bytes=usage_after.ru_maxrss*1024), indent=2)+'\n')
    if experiment and os.environ.get('NEUTRINO_ABLATION_STATS') == '1':
        assert experiment.neutrino_ablation_flush(str(out/'measurement-ablation.json').encode()) == 0
    summary = getattr(experiment, 'neutrino_capture_summary', None) if experiment else None
    if summary:
        summary.argtypes = [C.c_char_p]
        assert summary(str(out/'capture-summary.json').encode()) == 0
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
