# Reproduction coverage

The release keeps three separate questions explicit: restoring exact records,
recomputing retained measurements, and collecting new hardware measurements.
The original submission snapshot remains privately archived; the public release
contains the code needed for the paper's mechanisms and supplementary paths.

| Experiment | Implementation | How to run / requirements |
|---|---|---|
| Primary full inference | `scripts/reproduce.py`, `configs/conditions.json` | [Main recipe](REPRODUCE.md); Linux CUDA, licensed GGUF, pinned llama.cpp; optional nvCOMP |
| Baselines and reference ablation | `native serial async nvcomp ours sm24 sm32` | Same task, pool sizes and request endpoint; use rotated process rounds |
| B1–B16 and length sensitivity | `experiments/b16/llama_adapter_batch16.cpp`, `workload.py` | Builder creates the B16 adapter; `scripts/reproduce.py --batch-size 16 --context 8192 …` selects it |
| Document question answering | `experiments/document-qa/` | [Case guide](DOCUMENT_QA.md); real document prompts and complete inference, not a single operator |
| NVBit portability | `experiments/nvbit/` | [Frontend guide](FRONTENDS.md); external NVBit installation, real SASS producer |
| XTrace sparse / ERE | `experiments/xtrace/` | [Frontend guide](FRONTENDS.md); external GWatch/XTrace runtime and PTX replacement producer |
| Variable-event capacity prototype | `benchmarks/core_capacity_experiment.py`, `run_certified_core.py` | [Commands](REPRODUCE.md); separate supported PTX subset |
| Jetson source | `experiments/jetson/` | Historical cross-device source; primary release CUDA codec/hook can be built with `--arch 87` |
| Paper tables | `scripts/summarize_paper.py`, `results/paper/` | CPU only, no model or GPU |
| Exact recovery | `tests/`, `benchmarks/gpu_*_smoke.py` | CPU + CUDA boundary/fallback checks; correctness runs are separate from timed comparisons |

## Portability changes in v1.0.0

The `neutrino` import name and `NEUTRINO_*` variables remain compatible with the
measured implementation. `eretrace` is the new public CLI alias. The root runtime
is reconstructed byte-for-byte from the dependency manifest. B16 workload imports
now use checkout-relative paths; its adapter and algorithm are unchanged.
The document case reads `CASE_MODEL` rather than a particular machine's path.
The public reproduction wrapper's B16 selection is new orchestration, not a new
measurement or a change to the encoding algorithm.

Some experiment scripts are the preserved historical orchestration (including
SSH host and build-path references). They are source records, not the portable
entry point. Use the guides and public wrappers above for a newly provisioned
host; do not run old orchestration against somebody else's machine.

## Comparison contract

Start timing when each complete inference request starts. End after GPU work,
collection, required encoding checks/raw recovery, D2H, drain and trace file
close. Report file bytes at that same endpoint. Loading/JIT/warmup are explicit
setup; optional offline decoding, extra SHA/oracle validation and analysis are
separate. Do not infer a fresh result by subtracting stages from an older run.
Preserve all valid process rounds and record source/build/model hashes.
