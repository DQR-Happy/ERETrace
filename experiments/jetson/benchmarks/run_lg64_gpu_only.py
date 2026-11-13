"""GPU-only Activity diagnostic after the already completed main and stall suites."""
import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
primary=Path('/root/autodl-tmp/neutrino-lg64-detail-20261003')
assert (primary/'results/tail-complete.json').exists()
assert (root/'results/stall-3round/complete.json').exists()
out=root/'results';py=sys.executable
assert not (out/'gpu-only-3round').exists(), 'Refuse to overwrite completed or partial evidence'
env=dict(os.environ,PATH='/usr/local/cuda/bin:'+os.environ.get('PATH',''),
    NEUTRINO_COMPILER_CACHE_ROOT='/root/autodl-tmp/neutrino-ablation-20261002/compiler-cache')
cuda = '/usr/local/cuda/targets/x86_64-linux'
subprocess.run(['cc', '-shared', '-fPIC', '-O3', 'benchmarks/deep_cupti_gpuonly.c',
    '-I' + cuda + '/include', '-L' + cuda + '/lib', '-Wl,-rpath,' + cuda + '/lib',
    '-lcupti', '-ldl', '-lpthread', '-o', 'benchmarks/deep_cupti_gpuonly.so'], cwd=root, env=env, check=True)
profile = json.loads((primary / 'results/profile-specs.json').read_text())
for c in ['N', 'G', 'X', 'Y', 'Z']:
    profile[c]['root'] = str(primary)
    profile[c]['env'].update(DEEP_CUPTI_CALLBACKS='0',
        DEEP_CUPTI_LIBRARY=str(root / 'benchmarks/deep_cupti_gpuonly.so'))
ps = out / 'gpu-only-specs.json'
ps.write_text(json.dumps(profile, indent=2) + '\n')
subprocess.run([py, 'benchmarks/run_bulk_matrix.py', '--output', str(out / 'gpu-only-3round'),
    '--trace-root', '/dev/shm/neutrino-lg64-detail', '--spec', str(ps),
    '--conditions', 'N', 'G', 'X', 'Y', 'Z', '--rounds', '3', '--compiler-python', str(root / 'benchmarks/cache_python.py'),
    '--', py, str(root / 'benchmarks/qwen_lg64_api_workload.py'),
    '--adapter', '/root/autodl-tmp/xtrace-3090-repro/libqwen_adapter.so',
    '--backend', '/root/autodl-tmp/qwen3-32b-aiperf/llama.cpp/build/bin',
    '--model', '/autodl-fs/data/qwen3-small-aiperf/models/Qwen3-0.6B-Q4_K_M.gguf',
    '--requests', '1', '--warmup', '2', '--output-tokens', '32'], cwd=root, env=env, check=True)
(out/'gpu-only-complete.json').write_text(json.dumps(dict(complete=True))+'\n')
