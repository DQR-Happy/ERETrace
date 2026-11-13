"""Run native / passthrough / capture / capture+timing in rotated order.

Workload writes workload.json with wall_ns and output_digest to NEUTRINO_BENCH_DIR.
All conditions use the same binary, input and application synchronization.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

CONDITIONS = ('native', 'passthrough', 'capture', 'capture_timed', 'capture_sampled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--probe', default='block_sched')
    parser.add_argument('--kernel')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--keep-raw', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS[:4]))
    parser.add_argument('--sample-every', type=int, default=16,
                        help='sampling interval for the optional capture_sampled condition')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.repetitions < 1 or args.sample_every < 1:
        parser.error('a workload command and positive repetition count are required')
    if os.environ.get('NEUTRINO_HOOK_DRIVER'):
        parser.error('run the matrix outside a Neutrino-injected process')
    args.output = args.output.resolve()
    args.trace_root = args.trace_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.trace_root.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = dict(command=command, probe=args.probe, kernel=args.kernel,
                    repetitions=args.repetitions, conditions=args.conditions,
                    trace_filesystem=str(args.trace_root), python=sys.executable,
                    callback='/dev/null', timing_sample_every=1, timing_flush_every=0,
                    sampled_condition_interval=args.sample_every)
    (args.output / 'config.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for repeat in range(args.repetitions):
        order = args.conditions[repeat % len(args.conditions):] + args.conditions[:repeat % len(args.conditions)]
        for condition in order:
            run = args.output / f'r{repeat + 1}-{condition}'
            run.mkdir()
            trace = args.trace_root / f'{args.output.name}-r{repeat + 1}-{condition}'
            trace.mkdir(exist_ok=False)
            env = os.environ.copy()
            env['NEUTRINO_BENCH_DIR'] = str(run)
            argv = command
            if condition != 'native':
                argv = [sys.executable, '-m', 'neutrino.cli', '-p', args.probe,
                        '--callback', '/dev/null', '--tracedir', str(trace)]
                if condition in ('passthrough', 'capture_timed', 'capture_sampled'):
                    argv += ['--timing', '--timing-dir', str(run)]
                if condition == 'capture_sampled':
                    argv += ['--timing-sample-every', str(args.sample_every)]
                if condition == 'passthrough':
                    argv += ['--capture-mode', 'passthrough']
                if args.kernel:
                    argv += ['--kernel', args.kernel]
                argv += command
            (run / 'command.json').write_text(json.dumps(argv, indent=2) + '\n')
            print(f'r{repeat + 1} {condition}', flush=True)
            started = time.perf_counter_ns()
            try:
                with (run / 'stdout.log').open('wb') as stdout, (run / 'stderr.log').open('wb') as stderr:
                    process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr,
                                               start_new_session=True)
                    try:
                        rc = process.wait(timeout=args.timeout)
                        if rc:
                            raise subprocess.CalledProcessError(rc, argv)
                    except BaseException:
                        # The CLI has its own child; stop the entire experiment process group.
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=5)
                        except ProcessLookupError:
                            pass
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                wall = time.perf_counter_ns() - started
                workload = json.loads((run / 'workload.json').read_text())
                # Require proof of real capture, not a silent original-kernel fallback.
                stage_path = run / 'measurement-timing.json'
                if condition in ('capture_timed', 'capture_sampled'):
                    stage = json.loads(stage_path.read_text())
                    groups = stage['groups']
                    if not any(g['outcome'] == 'captured' and g['calls'] for g in groups):
                        raise RuntimeError('No measured capture launches')
                    if any(g['errors'] for g in groups):
                        raise RuntimeError('CUDA or trace-write errors in measured capture')
                counts = dict(files=0, bytes=0)
                for path in trace.glob('*/result/*.bin'):
                    counts['files'] += 1; counts['bytes'] += path.stat().st_size
                # Retain compact cold compiler phase data, never source/binary/raw trace payloads.
                compiler = []
                for path in trace.glob('*/kernel/*/process-timing.json'):
                    compiler.append(dict(kernel_directory=path.parent.name, **json.loads(path.read_text())))
                (run / 'compiler-timing.json').write_text(json.dumps(compiler, indent=2) + '\n')
                record = dict(repeat=repeat + 1, condition=condition, process_wall_ns=wall,
                              trace=counts, **workload)
                records.append(record)
                (run / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
            finally:
                if not args.keep_raw:
                    shutil.rmtree(trace)
            (args.output / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
    digests = {r['output_digest'] for r in records}
    if len(digests) != 1:
        raise RuntimeError('Output correctness differs across conditions or repetitions')
    by_mode = defaultdict(list)
    for record in records:
        by_mode[record['condition']].append(record['wall_ns'])
    medians = {mode: statistics.median(values) for mode, values in by_mode.items()}
    summary = dict(output_digests_match=True, median_wall_ns=medians,
                   wall_ns_ranges={k: [min(v), max(v)] for k, v in by_mode.items()})
    if 'native' in medians:
        summary['latency_increase_pct_vs_native'] = {k: (v / medians['native'] - 1) * 100 for k, v in medians.items()}
        summary['throughput_drop_pct_vs_native'] = {k: (1 - medians['native'] / v) * 100 for k, v in medians.items()}
    if {'capture', 'capture_timed'} <= medians.keys():
        summary['timer_delta_pct'] = (medians['capture_timed'] / medians['capture'] - 1) * 100
    if {'capture', 'capture_sampled'} <= medians.keys():
        summary['sampled_timer_delta_pct'] = (medians['capture_sampled'] / medians['capture'] - 1) * 100
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
