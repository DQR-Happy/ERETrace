#!/usr/bin/env python3
"""Audit the lossless multi-kernel NVBit container without changing its contents."""
import argparse
import collections
import json
from pathlib import Path
import re
import struct
import sys

HEADER = re.compile(rb"FRAME launch=(\d+) records=(\d+) bytes=(\d+) format=(raw|enc|nvc) flags=(\d+) grid=(\d+),(\d+),(\d+) block=(\d+),(\d+),(\d+) function=(\S+)\n")


def audit(path, check_fields, codec_path, oracle_dir):
    decode_flex = None
    if check_fields:
        sys.path.insert(0, str(codec_path))
        from trace_codec import decode_flex
    counts = collections.Counter()
    payload_bytes = collections.Counter()
    fallbacks = collections.Counter()
    functions = collections.Counter()
    total_records = 0
    with path.open("rb") as file:
        assert file.read(5) == b"NVB1\n"
        while header := file.readline():
            match = HEADER.fullmatch(header)
            if not match:
                raise ValueError(f"bad header at {file.tell()}: {header[:120]!r}")
            launch, records, size, fmt, flags, gx, gy, gz, bx, by, bz, function = match.groups()
            launch, records, size, flags, gx, gy, gz, bx, by, bz = map(int, (launch, records, size, flags, gx, gy, gz, bx, by, bz))
            fmt = fmt.decode()
            if launch != sum(counts.values()):
                raise ValueError(f"launch gap: {launch}")
            if records != gx * gy * gz * ((bx * by * bz + 31) // 32):
                raise ValueError(f"record count mismatch: {launch}")
            if fmt == "raw" and size != 16 * records:
                raise ValueError(f"raw byte count mismatch: {launch}")
            if check_fields and fmt in ("raw", "enc"):
                payload = file.read(size)
                if len(payload) != size:
                    raise ValueError(f"truncated payload: {launch}")
                raw = payload if fmt == "raw" else decode_flex(payload, records)
                if len(raw) != records * 16:
                    raise ValueError(f"decoded byte count mismatch: {launch}")
                if fmt == "enc" and oracle_dir is not None:
                    oracle = (oracle_dir / f"nvbit-{launch}.oracle.raw").read_bytes()
                    if raw != oracle:
                        raise ValueError(f"same-execution oracle mismatch: {launch}")
                for start, elapsed, smid in struct.iter_unpack("<QII", raw):
                    if not start or not elapsed or smid >= 82:
                        raise ValueError(f"invalid record in launch: {launch}")
            else:
                file.seek(size, 1)
                if file.tell() > path.stat().st_size:
                    raise ValueError(f"truncated payload: {launch}")
            counts[fmt] += 1
            payload_bytes[fmt] += size
            fallbacks[flags] += 1
            functions[function.decode()] += 1
            total_records += records
        if file.tell() != path.stat().st_size:
            raise ValueError("trailing or missing bytes")
    return {"path": str(path), "container_bytes": path.stat().st_size,
            "launches": sum(counts.values()), "records": total_records,
            "formats": dict(counts), "payload_bytes": dict(payload_bytes),
            "fallback_flags": dict(fallbacks), "unique_functions": len(functions),
            "field_check": check_fields,
            "same_execution_oracle": oracle_dir is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check-fields", action="store_true")
    parser.add_argument("--codec-path", type=Path, default=Path('/root/autodl-tmp/neutrino-review-sm-only-20261009/source/benchmarks'))
    parser.add_argument("--oracle-dir", type=Path)
    a = parser.parse_args()
    if a.oracle_dir and not a.check_fields:
        parser.error("--oracle-dir requires --check-fields")
    print(json.dumps(audit(a.path, a.check_fields, a.codec_path, a.oracle_dir), indent=2))
