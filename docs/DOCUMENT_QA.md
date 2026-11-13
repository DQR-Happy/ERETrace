# Document question-answering case

This case profiles complete real-text prefill and 64-token greedy generation,
with Flash Attention enabled in the adapter. It retains every captured record
and adds launch/phase sidecars for diagnosing the inference trace.

## Sources

- `experiments/document-qa/runtime/`: exact case runtime and encoder; bootstrap restores inherited frontend files.
- `experiments/document-qa/code/`: final workload, prompts, frame reader, analysis and independent validation.
- `experiments/document-qa/analysis/`: diagnostic probes, native CUPTI comparison, layout plotting and decoder source.
- `results/document-qa/`: retained per-request scalar inputs and stage breakdown.
- `results/paper/case-portability-results.json`: the final manuscript's case and frontend aggregates.

## Provision and run

Follow [INSTALL.md](INSTALL.md), including pinned llama.cpp and the optional
nvCOMP SDK. Install `pip install -e '.[case,plots]'`. Build the case with:

```bash
python scripts/case.py build --llama-source external/llama.cpp \
  --llama-build external/llama.cpp/build --arch 86
python scripts/case.py run --model models/Qwen3-8B-Q4_K_M.gguf \
  --backend external/llama.cpp/build/bin --condition ours \
  --output outputs/document-qa/ours --warmup 2 --requests 1
```

Use a **new output directory** for each condition/process. Choices are `native`,
`serial`, `async`, `nvcomp`, `ours`; nvCOMP additionally needs `case.py build
--nvcomp-root …`. `--dry-run` prints the resolved command without loading a GPU.
The workload stores complete closed traces; provision storage for the raw
condition (over 6 GB per measured request in the retained case). Do not put a
long comparison on a nearly full tmpfs. Native capture does not create a trace.

Run the same commands in balanced/seeded process order for each prompt index
(`--prompt-index 0`, `1`, `2`), with fixed model, backend, CPU quota and pools.
Fresh repetitions are new measurements, not the retained four-block paper data.
The wrapper disables optional offline analysis in performance runs; the original
analysis/diagnostic sources remain available for independent follow-up.

## Endpoint and retained example

`capture_ns` runs from real-text request start through trace closure. `prefill_ns`
and `drain_ns` are additive stages; generation is `capture - prefill - drain`.
Offline `analysis_ns` is not part of the paper's capture metric.
The retained case means are 1,808.42 ms / 2.730 GB for nvCOMP and 1,295.48 ms /
1.778 GB for ERETrace (four paired blocks). These are historical measurements,
not release smoke-test results. See the scalar data and the paper for scope.
