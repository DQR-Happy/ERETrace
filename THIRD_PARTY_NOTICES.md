# Third-party notices

ERETrace's author-owned additions, patches, documentation and artwork are offered
under the [MIT License](LICENSE). This grant does not relicense dependencies.
Original attribution in source headers is retained.

| Component | Source / terms | Distribution in v1.0.0 |
|---|---|---|
| Neutrino | [open-neutrino/neutrino](https://github.com/open-neutrino/neutrino), Songlin Huang and Chenshu Wu; pinned commit `4a82cd22f474c31ac2fecfa174d381a19bb3f469` | External dependency. Our patches are distributed; original/modified upstream files are materialized locally by `scripts/bootstrap.py`. |
| uthash | Troy D. Hanson; BSD-style notice embedded in `neutrino/src/uthash.h` and experiment copies | Source headers retain their complete notices. |
| SHA-1 helper | Brad Conte, [crypto-algorithms](https://github.com/B-Con/crypto-algorithms); upstream README dedicates the code to the public domain | Attribution retained in `sha1.h`; used for filenames, not security. |
| llama.cpp / ggml | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT | External pinned checkout; neither source archive nor model weights are bundled. |
| NVIDIA CUDA / CUPTI / nvCOMP / NVBit | NVIDIA; each SDK's own terms | External installations. Integration code is included; vendor headers, libraries and binaries are not redistributed. |
| NVBit sample Makefile | NVIDIA, BSD-3-Clause notice embedded in the file | Adapted build recipe in `experiments/nvbit/tool/Makefile`; full notice retained. |
| GWatch / XTrace | Tool provider's terms | External runtime; our collector, producer and experiment integration are included. |
| Python dependencies | NumPy (BSD), toml (MIT), optional matplotlib (PSF-based), zstandard (BSD) | Installed separately; no dependency wheels are vendored. |
| Model weights | Respective model/quantization providers | External downloads. Consult each model's terms and verify `configs/models.json`. |

## Neutrino dependency boundary

The pinned upstream `setup.py` declares an MIT classifier, but the repository
has no root LICENSE text. ERETrace does not manufacture that missing notice or
apply its own MIT grant to upstream code. Bootstrap obtains the original files
directly from the named upstream repository and applies our published patches;
`provenance/upstream-runtime.json` checks every base and resulting SHA-256.
This keeps the release's own license scope explicit while retaining the exact
runtime used by the implementation. A locally assembled runtime remains subject
to its constituent components' terms.

## Attribution

Neutrino is the programmable probing frontend on which the primary ERETrace
implementation is built. Its OSDI 2025 research and authorship remain distinct
from our execution-reference encoding and collection contributions. Please cite
both papers when building on both systems; see `CITATION.cff` and `docs/CITATION.md`.
