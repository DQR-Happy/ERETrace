# Evidence inventory

The files below are **retained measurements**, not new executions of the packaged workflow. The original authoritative evidence remains outside this repository and is unchanged. Copies strip machine-specific absolute location prefixes from metadata; numeric values are preserved.

| File | Paper role |
|---|---|
| [main-results.json](../results/paper/main-results.json) | Main three-model latency/volume comparison, length and batch sensitivity, application phases |
| [main-recovery.json](../results/paper/main-recovery.json) | Recovery Map counts, reason bits, weighted raw bytes and direct-raw small Maps |
| [reference-control.json](../results/paper/reference-control.json) | Same-pipeline CTA versus SM reference control and exact-restoration evidence |
| [discussion-results.json](../results/paper/discussion-results.json) | GPU topology, CPU encoding, local-vs-streaming and compaction boundaries |
| [appendix-results.json](../results/paper/appendix-results.json) | Extended correctness, direct capture, historical ablations, identities and source index |
| [main-environment.json](../results/paper/main-environment.json) | Confirmed primary host hardware |

Regenerate primary tables:

```bash
python scripts/summarize_paper.py --output outputs/paper-summary
```

The primary comparison uses 256 input tokens, 64 output tokens, B=1 and context 4096 on the RTX 3090. The confirmed host has 15 vCPU Intel Xeon Platinum 8358P @2.60GHz and 90GB RAM; the benchmark uses eight inference threads. The GPU has 24GB VRAM. Earlier development experiments used other device capacities and must retain their own environment labels.

Latency is completion-inclusive. File sizes use decimal GB; pool capacity uses MiB. Native has no trace file. Recomputed tables keep full-precision values in CSV and round only the Markdown display. `main-results.json` retains historical memory fields with their individual measurement scope; they are not combined into a single peak-memory comparison here.

## What this package can regenerate

- From scalar JSON: primary tables and recovery summaries.
- From source plus Linux GPU/models: primary fixed-record performance, SM-reference control, diagnostics and encoding oracle tests.
- From the original experiment index: identify distinct builds used for single-operator capture, variable-event experiments and historical sensitivity results.

A complete GPU rerun requires dependency versions, model identities and capture coverage checks on the target machine. This local packaging pass does not claim to re-execute historical GPU experiments.
