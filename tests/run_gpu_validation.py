"""Opt-in CUDA integration checks; run from the repository root on an idle GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--trace-root', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
args.trace_root = args.trace_root.resolve()
args.trace_root.mkdir(parents=True, exist_ok=True)
cases = {
    'fixed': [],
    'dynamic': ['-p', 'dmat'],
    'filtered': ['--kernel', 'intentionally_no_such_kernel'],
    'callback': ['--callback', str(root / 'tests/callback_smoke.py')],
    'sampled': ['--timing-sample-every', '3'],
    'passthrough': ['--capture-mode', 'passthrough'],
    'benchmark': ['--benchmark'],
    'memusage': ['--memusage'],
}
summary = []
for name, flags in cases.items():
    print(name, flush=True)
    output = args.output / name
    output.mkdir()
    trace = args.trace_root / (args.output.name + '-' + name)
    trace.mkdir(exist_ok=False)
    env = dict(os.environ, NEUTRINO_BENCH_DIR=str(output))
    command = [sys.executable, '-m', 'neutrino.cli', '--timing', '-p', 'block_sched',
               '--callback', '/dev/null', '--kernel', 'neutrino_timing_vector',
               '--timing-dir', str(output), '--tracedir', str(trace), *flags,
               sys.executable, str(root / 'benchmarks/triton_workload.py'),
               '--loops', '8', '--warmup', '2']
    try:
        with (output / 'stderr.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log,
                                       start_new_session=True)
            try:
                rc = process.wait(timeout=120)
                if rc:
                    raise subprocess.CalledProcessError(rc, command)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        workload = json.loads((output / 'workload.json').read_text())
        doc = json.loads((output / 'measurement-timing.json').read_text())
        assert doc['calls_completed'] == 8, doc['calls_completed']
        assert sum(g['calls'] for g in doc['groups']) == 8
        assert all(g['host_wall']['total_ns'] == sum(s['total_ns'] for s in g['stages'].values())
                   for g in doc['groups'])
        assert all(not g['errors'] for g in doc['groups'])
        stages = {stage for g in doc['groups'] for stage in g['stages']}
        outcome = ('fallback' if name in ('filtered', 'memusage') else
                   'passthrough' if name == 'passthrough' else 'captured')
        assert all(g['outcome'] == outcome for g in doc['groups'])
        if name == 'dynamic':
            assert {'count_submit', 'count_d2h_and_wait', 'count_cpu_reduce'} <= stages
        if name == 'callback':
            assert 'callback_process' in stages
        if name == 'benchmark':
            assert 'legacy_benchmark_replay' in stages and 'trace_d2h' not in stages
        if name == 'passthrough':
            assert stages == {'cache_lookup', 'original_submit'}
            assert not any(g['gpu_probe_interval']['samples'] for g in doc['groups'])
        if name == 'sampled':
            assert 0 < doc['calls_sampled'] < 8
        else:
            assert doc['calls_sampled'] == 8
        trace_count = 0
        for path in trace.glob('*/result/*.bin'):
            trace_count += 1
            with path.open('rb') as file:
                gx, gy, gz, bx, by, bz, shared, sections = struct.unpack('<8I', file.read(32))
                expected = 32 + 16 * sections
                for _ in range(sections):
                    stride, warp_div, offset = struct.unpack('<IIQ', file.read(16))
                    assert offset == expected
                    expected += gx * gy * gz * ((bx * by * bz + warp_div - 1) // warp_div) * stride
                assert expected == path.stat().st_size
        if name in ('fixed', 'dynamic', 'callback', 'sampled'):
            assert trace_count == 10, trace_count
        else:
            assert trace_count == 0
        summary.append(dict(case=name, passed=True, measured_calls=8,
                            sampled_calls=doc['calls_sampled'], trace_files_checked=trace_count,
                            output_digest=workload['output_digest']))
    finally:
        shutil.rmtree(trace)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
assert len({r['output_digest'] for r in summary}) == 1
print('All CUDA cases passed')
