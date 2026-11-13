# Contributing

Use an editable installation after `python3 scripts/bootstrap.py`.
Run `python -m unittest discover -s tests -p 'test_*.py' -v` and
`python scripts/check_repository.py` before proposing a change.

Keep the `neutrino` Python namespace and environment variables compatible; the
public project and command are named ERETrace / `eretrace`. Keep code changes
separate from frozen paper measurements. Changes to a generated upstream file
must update its patch and final digest in `provenance/upstream-runtime.json`;
verify restoration from a clean checkout before submitting.

Capture optimizations must preserve every record's fields, precision, order and
ownership. Keep extra validation outside formal performance timing. Include
hardware, versions, scope, endpoint, baseline and complete samples for performance
claims. Do not commit model weights, raw traces, private logs or vendor SDKs.
