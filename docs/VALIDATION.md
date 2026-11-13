# v1.0.0 validation

Validated on 2026-11-14. Source identities, outcome groups and new CUDA binary
hashes are in [release-validation.json](../provenance/release-validation.json).

| Check | Environment | Outcome |
|---|---|---|
| Pinned dependency + patch reconstruction | Pinned upstream commit | All 91 resulting file SHA-256 values matched |
| CPU suite, including public B16/case wrappers | Isolated Python 3.12, NumPy 2.2.6 | 61 tests passed |
| Public CLI and retained-table generation | Same environment | `eretrace --help` and `summarize_paper.py` passed |
| Hook, preload and CUDA encoder build | Jetson Orin, AArch64, CUDA 12.6.85, sm_87 | Passed; new binary hashes recorded |
| Reference / flexible / hierarchical / fixed / batch-delta CUDA checks | Same Jetson; same-buffer oracle and range/fallback cases | 24 / 36 / 12 / 2 / 5 groups passed (79 total) |
| Original CPU suite on CUDA host | Jetson, Python 3.10.12, NumPy 1.23.5 | 58 tests passed |
| Project homepage | Desktop light/dark and 390×844 mobile | Reviewed; model selector and theme switch checked |

CUDA boundary/overflow/SM checks establish exact restoration; they are not
performance rankings. The RTX 3090 full-inference paper data remain in
`results/paper/` and are not presented as new release measurements. External
NVBit/GWatch integrations retain their own dependencies and experiment scope.

## Repeat the checks

```bash
python scripts/bootstrap.py
python -m pip install -e '.[test]'
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/check_repository.py
python scripts/summarize_paper.py
```

After `scripts/build.py` on a Linux CUDA machine:

```bash
mkdir -p outputs/correctness
for test in reference flex hier codec batch_delta; do
  python benchmarks/gpu_${test}_smoke.py build/capture_pack.cubin outputs/correctness/${test}.json
done
```

`python scripts/package.py --output /outside/ERETrace-v1.0.0.tar.gz` exports only
the public source/patch payload. Bootstrap restores dependencies. The read-only
repository checker checks local links, credential patterns, release hashes and
reconstructed runtime identities.
