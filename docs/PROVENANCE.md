# Source provenance and packaging changes

## Primary source selection

The fixed-record runtime is copied from the source snapshot retained by the 2026-10-09 reviewer-control experiment. The hook, arena, condition definitions, encoder and compact helpers are preserved byte for byte. The snapshot matched the frozen primary hook/encoder source at the time of that experiment.

The nvCOMP adapter is copied from the **parallel-layout** source in the final-closure `source-primary` snapshot, rather than the older serial-layout file present in the reviewer source directory. Its API parameters, layout and parallel prefix implementation are preserved. This is an identified source implementation; source identity does not manufacture a historical SDK version or binary identity.

[Source origins](../provenance/source-origins.json) records every imported file's workspace-relative source and SHA-256. `scripts/check_repository.py` checks the critical unchanged source hashes. Release archives additionally carry a hash for every packaged file.

## Packaging-layer edits

1. Replace the legacy auto-pip/auto-detection build script with an explicit Linux CUDA builder accepting toolkit, driver and architecture arguments. Add a compatibility `python -m neutrino.build` entry.
2. Retain the established matrix runner and workload; change the workload's import path to this checkout's `benchmarks/`.
3. Strengthen cross-condition output validation to compare the all-sequence digest when present, not only sequence zero.
4. Resolve CUDA libraries in GPU smoke tests through `CUDA_DRIVER_LIBRARY` or the system loader rather than a fixed x86 path.
5. Add portable condition selection, dry-run output, input hashes, paired summaries, packaging and CPU CI. Those are new artifact utilities, not historical timed binaries.
6. Copy scalar paper data with machine-specific absolute location prefixes normalized; preserve numeric results and source hashes.
7. Add exact static-analysis/certificate modules from the retained capacity-recheck snapshot and their CPU tests. Keep this prototype separate in the documentation.

## Upstream lineage

The original local Neutrino checkout identifies upstream commit `4a82cd22f474c31ac2fecfa174d381a19bb3f469` and remote `https://github.com/open-neutrino/neutrino`. The research snapshots include substantial local changes and are not represented as that unmodified commit. No private `.git` history, credentials, generated cubin, model, or raw trace is copied into this artifact. Inherited frontend files are fetched at bootstrap and checked against the pinned base/result hashes; only our modifications are published as patches.

## Version boundaries

The portable primary adapter is the exact-length B≤8 reviewer-control snapshot.
The actual B16 adapter and driver are available under `experiments/b16/`; the public runner selects that version for B=16.
The original llama.cpp source identity was subsequently resolved to
`81ff93ea1d48c0482508d30e75b814df99c25bf0`; see
`provenance/llama-source-recovery.json`. New builds retain their own identities.

The version audit reconstructs nvCOMP package 5.3.0.16 and the pinned llama.cpp commit from the retained installation/source evidence. It also distinguishes early library hashes from the exact library loaded in primary runs. Record the installed version and loaded-library hash on every new run; neither package identity nor the adapter source replaces that record.

8. Let `neutrino --help` work before GPU compilation and provide a clear build instruction for actual invocations without a hook. CLI support points to this artifact's documentation.
9. Retain the later certified-capacity T/C driver, fixtures and helper source as an explicitly scoped experiment. They are distinct from the earlier capacity smoke fixture.
