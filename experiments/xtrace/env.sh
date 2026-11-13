#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/review1-3090-20261011
PAPER=/root/autodl-tmp/neutrino-review-sm-only-20261009
export GGML_CUDA_DISABLE_GRAPHS=1 NEUTRINO_BENCH_THREADS=8 NEUTRINO_BENCH_CONTEXT=4096
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:$ROOT/cupti128/nvidia/cuda_cupti/lib/libcupti.so.12
export LD_LIBRARY_PATH=/root/autodl-tmp/ispa-cpu-compat:$ROOT/cupti128/nvidia/cuda_cupti/lib:/usr/local/cuda/lib64
export PYTHONPATH=$ROOT/triton310:$ROOT/gwatch-python
export QWEN_ADAPTER=$PAPER/runs/3090-unblocked-20261006/build/libqwen_extended.so
export QWEN_MODEL="${QWEN_MODEL:-$PAPER/models-official/Qwen3-8B-Q4_K_M.gguf}"
export QWEN_BACKEND=$PAPER/llama-jetson-build/bin
export XTRACE_CACHE=/tmp/xtrace-paper-gpu-cache
export XTRACE_GPU_LIB=/tmp/xtrace_gpu_collector_unit32.so
exec "$ROOT/proton-env/bin/python" /tmp/xtrace-paper-gpu-nosys/source/run.py "$@"
