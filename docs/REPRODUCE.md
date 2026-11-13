# Reproduction guide

## Three independent entry points

1. **Recompute retained numbers:** `python scripts/summarize_paper.py`. Reads local JSON and writes new CSV/Markdown. No GPU, weights, or network.
2. **Check implementation correctness:** CPU unit tests, followed by CUDA encoding and capacity smoke tests.
3. **Collect fresh performance:** `scripts/reproduce.py`. Independent processes, fixed input/output lengths, rotated order, matched model, native output digest, and completion-inclusive request timing.

All generated results go into a new `outputs/` directory; historical evidence is read-only input. A failed process remains a failed run. The runner stops on unequal output digests and never searches for a favorable stopping point.

## Primary full-request experiment

Build following [INSTALL.md](INSTALL.md). Start with a small licensed model and `--rounds 1 --requests 1` for a smoke check, then run the full comparison on each target model:

```bash
python scripts/reproduce.py \
  --model models/Qwen3-8B-Q4_K_M.gguf \
  --backend external/llama.cpp/build/bin \
  --conditions native serial async nvcomp ours \
  --input-tokens 256 --output-tokens 64 --context 4096 --batch-size 1 \
  --threads 8 --warmup 2 --requests 3 --rounds 7 \
  --output outputs/qwen8b-main
```

`nvcomp` requires its optional build. Omit that condition if the SDK is unavailable, while labeling the smaller comparison explicitly. Every condition uses the same inference adapter. Inputs are deterministic exact-length valid token sequences; outputs are actual greedy tokens, including after EOS until the configured count. First-token selection uses prefill logits; the remaining output tokens invoke decode. This is an in-process controlled workload, not an HTTP throughput test.

| Condition | Meaning |
|---|---|
| `native` | No Neutrino hook loaded |
| `serial` | Retained Neutrino serial file-capture path, with the compatibility fixes in the source snapshot |
| `async` | Uncompressed fixed-slot asynchronous collection, 32 slots and separate copy stream |
| `nvcomp` | Same arena infrastructure, field-separated Cascaded and parallel layout |
| `ours` | CTA-reference encoding, nominal transfers, exact recovery, bounded arena and writev |
| `sm24`, `sm32` | Same pipeline with single-level SM-relative deltas of the named width |

The serial condition is the retained compatible implementation, not a claim that this artifact's hook is byte-identical to an untouched upstream release. NVBit and XTrace remain external tools; this package does not rebrand its capture paths as their implementations.

## Nearest-neighbor reference control

```bash
python scripts/reproduce.py \
  --model models/Qwen3-8B-Q4_K_M.gguf \
  --backend external/llama.cpp/build/bin \
  --conditions native ours sm24 sm32 \
  --input-tokens 256 --output-tokens 64 --context 4096 \
  --rounds 5 --output outputs/reference-control
```

`configs/conditions.json` records the exact per-condition environment. The wrapper fills only checkout locations and requested workload sizes. Run `--dry-run` to inspect the full command and environment. Use `--diagnostic` in a separate output directory to collect stages/recovery counters; the main measurement leaves instrumentation disabled.

## Length and batch sensitivity

Hold context at 8192. Run the four `(input, output)` pairs `(128,32)`, `(2048,32)`, `(128,256)`, `(2048,256)` with B=1 and `native nvcomp ours`. For batch sensitivity, hold 256/64 and vary B=1,2,4,8. Each cell needs its own output directory and complete paired rounds.

B=1/2/4/8 use the retained primary adapter; B=16 selects `experiments/b16/workload.py` and `build/libllama_adapter_b16.so`. The builder creates both adapters; their source identities are separately recorded. GPU memory, aggregate KV capacity, and trace storage must be checked before larger cases.

## Measurement and output

The application completes model synchronization, capture drain, and file close **before** each request timer ends. Stored latency is the sum of these request intervals. Between requests, the harness validates and deletes closed trace files; this offline validation/cleanup time is retained separately in `measurement_window_ns`. No `fsync` is added: default storage is tmpfs, and close is not a durable-storage guarantee.

Typical output:

```text
config.json           Conditions, rotated order and workload command
r1-ours/run.json       Process/request metrics, digests and structural checks
r1-ours/workload.json  Prefill/decode/drain breakdown and all-sequence digests
r1-ours/capture-summary.json  Runtime bytes, recovery and ownership counters
r1-ours/stderr.log     Local diagnostic log, excluded from release archives
runs.json             All successful process records
complete.json         Completion marker for the whole matrix
environment.json      Input/build hashes and runtime environment
summary.csv/json      Per-condition round means and paired native differences
```

The statistical unit is a process round. Seven rounds with three requests each are seven independent process observations, not 21 independent experiments. The summary uses paired differences and seeded 10,000-resample bootstrap intervals. Preserve all rounds. Single-round smoke intervals are not statistical evidence.

## Correctness tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
mkdir -p outputs/correctness
python benchmarks/gpu_reference_smoke.py build/capture_pack.cubin \
  outputs/correctness/reference.json
python benchmarks/gpu_flex_smoke.py build/capture_pack.cubin \
  outputs/correctness/flat.json
```

The CUDA tests check synthetic boundary values and exact reconstruction; run on a freshly built architecture. CPU tests cover modulo integer restoration, fallback flags, malformed payloads, capacity analysis, accounting, and live-interval allocation. A digest check establishes model-output consistency, while record restoration tests establish format correctness; both are needed.

For the independent event-capacity prototype:

```bash
PYTHONPATH=. python benchmarks/core_capacity_experiment.py \
  --output outputs/capacity-smoke --arch sm_86 --build --smoke
```

This retained prototype expects a CUDA toolkit at `/usr/local/cuda` and GNU `g++` for its small helper. It uses generated irregular PTX fixtures; it is not a general replacement for the primary LLM runtime.

## Historical coverage versus this portable entry point

The retained paper JSON also includes direct single-operator comparisons, fault injection, B16, topology limits, and ablations from independently frozen builds. Their numbers can be inspected here, but the primary runner does not recreate all of those builds. [RESULTS.md](RESULTS.md) maps the evidence, and [PROVENANCE.md](PROVENANCE.md) records the exact source snapshots. Direct single-operator capture and streaming full-inference capture are distinct paths; never relabel one as the other.

### Certified capacity-class versus per-thread-table control

The later Table 4 recheck source and its two real operator PTX inputs are also retained. This driver is deliberately scoped to its original RTX 3090 / `sm_86` environment:

```bash
PYTHONPATH=. python benchmarks/run_certified_core.py \
  --inputs fixtures/ptx --output outputs/certified-capacity \
  --build --smoke --rounds 1 --modes T C
```

Remove `--smoke` and choose the recorded case/round counts for a full capacity comparison. The runner creates isolated program instances, checks certificates and event counts, and alternates table/class modes. Its fixed `sm_86` compile target is part of the retained experimental source; it is not a cross-device implementation claim.
