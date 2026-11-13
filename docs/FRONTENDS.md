# NVBit and XTrace integration

These are separate frontend experiments. The primary Neutrino collector is not
relabeled as native NVBit or XTrace. Obtain and install each external tool under
its own terms; no proprietary SDK is shipped in this repository.

## NVBit: SASS instrumentation

`experiments/nvbit/tool/instr_count_matrix.cu` implements the bounded asynchronous
Raw / nvCOMP / ERE collector. `inject_funcs.cu` is the injected producer. The
included BSD-3-Clause SDK sample Makefile is adapted to compile only this final
collector, with C++17 and the injected-function flags. It does not link the older
`instr_count.cu` experiment into the same library.

Obtain NVBit from its [official releases](https://github.com/NVlabs/NVBit/releases)
and set `NVBIT_ROOT` to the extracted SDK directory. CUDA tools, GNU Make, a C++
compiler and `bc` must be on PATH. The SDK must contain `core/libnvbit.a` and the
matching headers; retain its version with your results.

```bash
export NVBIT_ROOT="$PWD/external/nvbit"
export PATH="/usr/local/cuda/bin:$PATH"
make -C experiments/nvbit/tool NVBIT_PATH="$NVBIT_ROOT/core" ARCH=sm_86
export NVBIT_TOOL="$PWD/experiments/nvbit/tool/eretrace_nvbit.so"
```

Build ERETrace's encoder and optional nvCOMP adapter using `scripts/build.py`.
With `NVBIT_TOOL` set to your built shared library and `MODEL`/`BACKEND` set to
the pinned model and llama.cpp build/bin directory, run one request:

```bash
mkdir -p outputs/nvbit/ours /dev/shm/eretrace-nvbit
export CUDA_INJECTION64_PATH="$NVBIT_TOOL"
export NVBIT_CAPTURE_ALL=1 NVBIT_STREAM=1 NVBIT_QUIET=1 ACTIVE_FROM_START=0
export NVBIT_ASYNC_SLOTS=4 NVBIT_MAX_WARPS=607744
export NVBIT_BACKEND=hybrid NVBIT_LOCAL_BITS=16 NVBIT_ELAPSED_BYTES=3
export NVBIT_ENCODER_CUBIN="$PWD/build/capture_pack.cubin"
export SCOPE_TRACE_DIR=/dev/shm/eretrace-nvbit
export REVIEW_NVBIT_PROFILE=1 REVIEW_NVBIT_WARMUP_PROFILE=1
export NEUTRINO_BENCH_DIR="$PWD/outputs/nvbit/ours"
export GGML_CUDA_DISABLE_GRAPHS=1 NEUTRINO_BENCH_CONTEXT=4096
export NEUTRINO_BENCH_STREAM_TRACE=1 NEUTRINO_BENCH_THREADS=8
python experiments/nvbit/qwen_exact_nvbit.py --adapter build/libllama_adapter.so \
  --backend "$BACKEND" --model "$MODEL" --input-tokens 256 \
  --output-tokens 64 --batch-size 1 --warmup 1 --requests 1
python experiments/nvbit/qwen_stream.py /dev/shm/eretrace-nvbit/trace-stream.nvb
```

Use fresh output/trace directories for every process. `NVBIT_BACKEND=raw` selects
the matched raw collector; `nvcomp` additionally needs `NVBIT_NVCOMP_LIB` pointing
to `build/nvcomp_pipeline.so`. Retain the trace before starting a subsequent run.
Historical final paired orchestration is in `qwen_final_compare.py`; its original
host paths are retained for provenance, not as the portable command.

## XTrace: PTX replacement, Sparse versus Sparse + ERE

`experiments/xtrace/producer.py` performs the PTX producer replacement through
GWatch/XTrace. `collector.cu` implements the matched sparse/ERE GPU collector.
This experiment is **not** an implementation of XTrace's native SASS rewriting
engine. Preserve that distinction when interpreting the paper's portability data.

In the Python environment in which GWatch, its native runtime and compatible
CUPTI/Triton components are installed:

```bash
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_86 -shared -Xcompiler=-fPIC \
  experiments/xtrace/collector.cu -o build/xtrace_collector.so -lcrypto -lpthread
export QWEN_ADAPTER="$PWD/build/libllama_adapter.so"
export QWEN_MODEL="$MODEL" QWEN_BACKEND="$BACKEND"
export XTRACE_GPU_LIB="$PWD/build/xtrace_collector.so"
export XTRACE_CACHE=/tmp/eretrace-xtrace-cache
export GGML_CUDA_DISABLE_GRAPHS=1 NEUTRINO_BENCH_THREADS=8 NEUTRINO_BENCH_CONTEXT=4096
python experiments/xtrace/run.py --mode sparse --out outputs/xtrace/sparse-r1
python experiments/xtrace/run.py --mode ere --out outputs/xtrace/ere-r1
```

The collector requires CUDA, a C++17 toolchain, OpenSSL development libraries
and pthreads. The runtime must export `gwatch.cuda.profile.ProfileContext`,
`gwatch.cuda.binary.Cubin`, and `gwatch.cuda.xtrace._runner.find_kernel_ptx`.
Use the runtime version matching those APIs, with visible embedded PTX. Keep
its package/commit identity with the run. `--verify` performs an independent
correctness run; keep it out of formal timing. `--retain-last` preserves the
last output stream. The preserved `env.sh` records the old installation layout
and is not a universal installer for an unavailable third-party runtime.

For a performance comparison, alternate or seed process order and use identical
warmup, model, scope, completion endpoint and resource budget. Do not treat a
single fixed-order invocation as a publishable ranking.
