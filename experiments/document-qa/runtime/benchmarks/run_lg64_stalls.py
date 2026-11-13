"""Small follow-up for FIFO wait owners and actual slot capacities.

Run from an isolated copy after run_lg64_tail.py; the primary binary stays frozen.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
primary = Path('/root/autodl-tmp/neutrino-lg64-detail-20261003')
assert (primary / 'results/tail-complete.json').exists(), 'Finish other GPU work first'
env = dict(os.environ, PATH='/usr/local/cuda/bin:' + os.environ.get('PATH', ''),
    NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
py = sys.executable
subprocess.run([py, 'neutrino/build.py'], cwd=root, env=env, check=True)
subprocess.run([py, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_capture_detail.py'],
    cwd=root, env=env, check=True)
out = root / 'results'
out.mkdir(exist_ok=True)
specs = json.loads((primary / 'results/specs.json').read_text())
specs['G']['root'] = str(primary)
sp = out / 'specs.json'
sp.write_text(json.dumps(specs, indent=2) + '\n')
subprocess.run([py, 'benchmarks/run_bulk_matrix.py', '--output', str(out / 'stall-3round'),
    '--trace-root', '/dev/shm/neutrino-lg64-detail', '--spec', str(sp),
    '--conditions', 'N', 'G', 'D', '--rounds', '3', '--compiler-python', str(root / 'benchmarks/cache_python.py'),
    '--', py, str(root / 'benchmarks/qwen_capture_workload.py'),
    '--adapter', '/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
    '--backend', '/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
    '--model', '/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
    '--requests', '3', '--warmup', '2', '--output-tokens', '32'], cwd=root, env=env, check=True)
(out / 'complete.json').write_text(json.dumps(dict(complete=True)) + '\n')
