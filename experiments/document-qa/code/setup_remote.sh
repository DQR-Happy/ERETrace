#!/usr/bin/env bash
set -euo pipefail
# Upload this directory with its code/, protocol.md and allocation.json first.
case_root="${1:-/root/ispa-e2e-noverify-20261013}"
original_case="${2:-/root/autodl-tmp/neutrino-inference-case-20261013}"
test -f "$original_case/code/reference_decode.so"
test -f "$original_case/source/benchmarks/nvcomp_pipeline.so"
test -f "$original_case/cache-python"
mkdir -p "$case_root/code"
for name in source backend; do
  if [[ ! -e "$case_root/$name" ]]; then ln -s "$original_case/$name" "$case_root/$name"; fi
  test "$(readlink "$case_root/$name")" = "$original_case/$name"
done
if [[ ! -e "$case_root/code/reference_decode.so" ]]; then
  ln -s "$original_case/code/reference_decode.so" "$case_root/code/reference_decode.so"
fi
/root/miniconda3/bin/python -m compileall -q "$case_root/code"
