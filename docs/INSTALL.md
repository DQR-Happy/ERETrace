# Installation

## Supported starting point

The paper's primary experiments used Linux x86_64, an RTX 3090, CUDA Toolkit 12.4.131, and Python 3.12. GPU execution is Linux-only. The build accepts explicit driver/toolkit paths and `--arch 86`, `87`, or `90`; changing architecture requires fresh correctness and performance validation. The release CUDA encoder/hook is also smoke-tested on Jetson Orin; see VALIDATION.md for the exact scope.

The documented Python environment pins `toml==0.10.2` and `numpy==2.2.6`. System prerequisites: Python 3.10–3.12 with venv, a C/C++ compiler, binutils (`nm`), CMake, Git, an NVIDIA driver, and CUDA's `nvcc`, `ptxas`, and `cuobjdump` on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python scripts/bootstrap.py
python -m pip install -e '.[test]'
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

Keep this checkout editable: runtime-generated hook files live in `neutrino/build/`, and the compiler generates `neutrino/src/signature.c` and `unmodified.c`. Those files are excluded from Git and release packages. Installing this distribution does not build or overwrite `/usr/lib` driver libraries.

## llama.cpp and the inference adapter

The original source archive was subsequently recovered and verified against
`81ff93ea1d48c0482508d30e75b814df99c25bf0`. See
`provenance/llama-source-recovery.json` and `provenance/software-version-audit.json`.
Historical measured binaries and new builds are separately identified by SHA-256.

```bash
mkdir -p external
# External dependency; this command downloads public upstream source.
git clone https://github.com/ggml-org/llama.cpp.git external/llama.cpp
git -C external/llama.cpp checkout 81ff93ea1d48c0482508d30e75b814df99c25bf0
cmake -S external/llama.cpp -B external/llama.cpp/build \
  -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF
cmake --build external/llama.cpp/build --parallel 8
python scripts/build.py --arch 86 \
  --llama-source external/llama.cpp \
  --llama-build external/llama.cpp/build
```

The hook needs embedded PTX for kernels it rewrites. Before a full experiment, inspect the CUDA backend with `cuobjdump --list-ptx external/llama.cpp/build/bin/libggml-cuda.so` and run a diagnostic smoke workload. Retain both native and virtual-code targets when configuring a different toolchain.

Use `--driver /absolute/path/to/libcuda.so.1` if driver detection fails, including on Jetson. Supply the installed driver, never CUDA's link-time stub. `build/build-manifest.json` records generated binaries and driver hashes. Preserve it with every experiment.

## Optional nvCOMP baseline

Obtain the nvCOMP SDK from NVIDIA under its license; place its headers and shared library under a local prefix. No nvCOMP binaries are redistributed here.

```bash
python scripts/build.py --arch 86 \
  --llama-source external/llama.cpp --llama-build external/llama.cpp/build \
  --nvcomp-root external/nvcomp
```

The paper version audit identifies nvCOMP package `5.3.0.16`; record your installed SDK and loaded-library hash. The prefix must expose `include/nvcomp/cascaded.h` and `lib/libnvcomp.so` (or `lib64/`). This builds the retained field-separated adapter with parallel prefix-layout code. It does not substitute the earlier serial-layout baseline.

Cascaded settings: `NVCOMP_TYPE_UINT`; four per-Map uint32 columns (timestamp low/high, elapsed, SM); 64 KiB outer chunks; 4096-byte internal chunks; two RLE layers, one delta layer, bitpacking enabled. GPU chunk lengths return to the writer before exact-size payload D2H. Transposition, scratch, metadata, and payload completion belong to the measured path. Record your installed SDK version and library hash; a historical exact SDK version must not be inferred from a source API alone.

## Models

Use legally obtained GGUF models: Qwen3-8B, Llama3.2-1B, and Gemma3-4B in the quantizations associated with your intended evidence set. Keep weights in `models/` or outside the checkout. Pass `--model-sha256` for identity enforcement. The runner hashes the actual file even without an expected checksum. Downloads, account credentials, and weights are deliberately separate from this artifact.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing `config.toml` | Run `scripts/build.py` after Python installation |
| PTX extraction failure | Check embedded PTX, supported kernel path, `cuobjdump` and `ptxas` on `PATH` |
| Backend load failure | Check `--backend`, shared ggml libraries and adapter rpath |
| Native works, capture fails | Run one request with `--diagnostic`; inspect `stderr.log` and stage JSON |
| Trace storage exhausted | Provision enough tmpfs for the largest single request; validation deletes files only after request completion |
| Unexpected output digest | Keep the failed run, stop the comparison, and inspect correctness before timing |
| Compiler rejects nvCOMP calls | Record installed SDK/API versions; use the compatible Cascaded API, rather than silently dropping baseline features |

Expected primary model filenames and SHA-256 values are provided in [configs/models.json](../configs/models.json). Use those identities for historical comparisons rather than selecting a different conversion with the same model name.
