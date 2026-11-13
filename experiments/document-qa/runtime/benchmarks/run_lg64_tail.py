"""Extra serial diagnostics, only after the main LG64 suite has completed."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
out = root / 'results'
assert (out / 'complete.json').exists(), 'Run the main suite first; do not overlap GPU measurements'
py = sys.executable
env = dict(os.environ, PATH='/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
    NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
dest = out / 'cabi-3round'
if not (dest / 'complete.json').exists():
    cmd = [py, 'benchmarks/run_bulk_matrix.py', '--output', str(dest),
        '--trace-root', '/dev/shm/neutrino-lg64-detail', '--spec', str(out / 'specs.json'),
        '--conditions', 'N', 'G', 'Z', '--rounds', '3', '--compiler-python', str(root / 'benchmarks/cache_python.py'),
        '--', py, str(root / 'benchmarks/qwen_lg64_api_workload.py'),
        '--adapter', '/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
        '--backend', '/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
        '--model', '/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
        '--requests', '3', '--warmup', '2', '--output-tokens', '32']
    subprocess.run(cmd, cwd=root, env=env, check=True)
dest = out / 'activity-only-3round'
if not (dest / 'complete.json').exists():
    cuda = '/usr/local/cuda/targets/x86_64-linux'
    subprocess.run(['cc', '-shared', '-fPIC', '-O3', 'benchmarks/deep_cupti_activity.c',
        '-I' + cuda + '/include', '-L' + cuda + '/lib', '-Wl,-rpath,' + cuda + '/lib',
        '-lcupti', '-ldl', '-lpthread', '-o', 'benchmarks/deep_cupti_activity.so'], cwd=root, env=env, check=True)
    specs = json.loads((out / 'profile-specs.json').read_text())
    for c in ['N', 'G', 'X', 'Y', 'Z']:
        specs[c]['env'].update(DEEP_CUPTI_CALLBACKS='0',
            DEEP_CUPTI_LIBRARY=str(root / 'benchmarks/deep_cupti_activity.so'))
    sp = out / 'activity-only-specs.json'
    sp.write_text(json.dumps(specs, indent=2) + '\n')
    cmd = [py, 'benchmarks/run_bulk_matrix.py', '--output', str(dest),
        '--trace-root', '/dev/shm/neutrino-lg64-detail', '--spec', str(sp),
        '--conditions', 'N', 'G', 'X', 'Y', 'Z', '--rounds', '3',
        '--compiler-python', str(root / 'benchmarks/cache_python.py'), '--', py,
        str(root / 'benchmarks/qwen_lg64_api_workload.py'),
        '--adapter', '/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
        '--backend', '/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
        '--model', '/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
        '--requests', '1', '--warmup', '2', '--output-tokens', '32']
    subprocess.run(cmd, cwd=root, env=env, check=True)
subprocess.run([py, 'benchmarks/run_lg64_syscalls.py'], cwd=root, env=env, check=True)
(out / 'tail-complete.json').write_text(json.dumps(dict(complete=True)) + '\n')
