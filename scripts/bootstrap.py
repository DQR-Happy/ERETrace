#!/usr/bin/env python3
"""Materialize the pinned Neutrino dependency and apply ERETrace's patches.

The public distribution contains our additions and patches. Unmodified upstream
files are obtained directly from upstream, with base and final SHA-256 checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bootstrap(upstream):
    spec = json.loads((ROOT / 'provenance/upstream-runtime.json').read_text())
    upstream = upstream.resolve()
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', spec['repository'],
                        str(upstream)], check=True)
    subprocess.run(['git', '-C', str(upstream), 'cat-file', '-e',
                    spec['commit'] + '^{commit}'], check=True)
    restored = {}
    with tempfile.TemporaryDirectory(prefix='eretrace-bootstrap-') as temp:
        stage = Path(temp)
        for name, entry in spec['files'].items():
            target = ROOT / name
            if target.exists() and sha(target.read_bytes()) != entry['sha256']:
                raise RuntimeError(f'Refusing to overwrite modified runtime: {name}')
            data = subprocess.check_output(['git', '-C', str(upstream), 'show',
                                            spec['commit'] + ':' + entry['base_path']])
            if sha(data) != entry['base_sha256']:
                raise RuntimeError(f'Upstream identity mismatch: {name}')
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if entry.get('patch'):
                subprocess.run(['git', 'apply', '--no-index', '--whitespace=nowarn',
                                str(ROOT / entry['patch'])], cwd=stage, check=True)
            final = dest.read_bytes()
            if sha(final) != entry['sha256']:
                raise RuntimeError(f'Patched identity mismatch: {name}')
            restored[name] = final
        # Verify every reconstruction before installing any runtime files.
        for name, data in restored.items():
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(json.dumps({'passed': True, 'upstream_commit': spec['commit'],
                      'runtime_files_verified': len(restored)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path,
                        default=ROOT / 'external/neutrino-upstream',
                        help='Existing upstream Git checkout, or clone destination')
    args = parser.parse_args()
    bootstrap(args.upstream)


if __name__ == '__main__':
    main()
