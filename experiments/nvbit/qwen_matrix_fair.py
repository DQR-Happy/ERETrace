#!/usr/bin/env python3
"""Corrected fair Qwen matrix: no inline per-record scan in any timed arm."""
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "qwen-matrix-fair"
REMOTE = "/root/autodl-tmp/review1-3090-20261011"
BACKEND = "/root/autodl-tmp/neutrino-review-sm-only-20261009/llama-jetson-build/bin"
MODEL = "/root/autodl-tmp/neutrino-review-sm-only-20261009/models-official/Qwen3-8B-Q4_K_M.gguf"
ADAPTER = "/root/autodl-tmp/neutrino-review-sm-only-20261009/runs/3090-unblocked-20261006/build/libqwen_extended.so"
ENCODER = "/root/autodl-tmp/neutrino-review-sm-only-20261009/source/benchmarks/capture_pack.cubin"
EXPECTED_DIGEST = "bd2a29b7618141b850f99a9edef5a359c479051557d970db24bd48f630e6168d"
METHODS = ("native", "raw", "wide16e3", "nvcomp")


def ssh(command, **kwargs):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "3090-exp", command], check=True, **kwargs)


def remote_run(round_id, method):
    name = f"r{round_id}-{method}"
    dest = f"{REMOTE}/qwen-matrix-fair/{name}"
    trace = f"/dev/shm/review1-qwen-matrix-fair-{name}"
    ssh(f"mkdir -p {shlex.quote(dest)} {shlex.quote(trace)}")
    env = {
        "PATH": "/usr/local/cuda/bin:$PATH",
        "NEUTRINO_BENCH_DIR": dest,
        "LD_LIBRARY_PATH": f"{BACKEND}:${{LD_LIBRARY_PATH:-}}",
    }
    if method != "native":
        env.update(REVIEW_NVBIT_PROFILE="1", REVIEW_NVBIT_WARMUP_PROFILE="1",
                   NVBIT_CAPTURE_ALL="1", NVBIT_STREAM="1", NVBIT_QUIET="1",
                   ACTIVE_FROM_START="0", NVBIT_MAX_WARPS="1048576",
                   SCOPE_TRACE_DIR=trace,
                   CUDA_INJECTION64_PATH=f"{REMOTE}/qwen_scope_fair/qwen_scope_fair.so")
        if method == "nvcomp":
            env.update(NVBIT_BACKEND="nvcomp", NVBIT_NVCOMP_LIB=f"{REMOTE}/nvcomp_pipeline.so")
        elif method == "raw":
            env["NVBIT_BACKEND"] = "raw"
        else:
            env.update(NVBIT_BACKEND="ours", NVBIT_LOCAL_BITS="16",
                       NVBIT_ELAPSED_BYTES="3", NVBIT_ENCODER_CUBIN=ENCODER)
    envs = " ".join(f"{k}={v if '$' in v else shlex.quote(v)}" for k, v in env.items())
    command = ["python3", "qwen_exact_nvbit.py", "--adapter", ADAPTER,
               "--backend", BACKEND, "--model", MODEL, "--warmup", "1",
               "--requests", "1", "--input-tokens", "256", "--output-tokens", "64",
               "--batch-size", "1"]
    shell = (f"cd {REMOTE} && env {envs} timeout 600 "
             + " ".join(shlex.quote(x) for x in command)
             + f" >{dest}/run.log 2>&1")
    ssh(shell)
    if method != "native":
        ssh(f"python3 {REMOTE}/qwen_stream.py {trace}/trace-stream.nvb >{dest}/audit.json")
    local_dir = LOCAL / name
    local_dir.mkdir(parents=True, exist_ok=True)
    for file in ("run.log", "workload.json") + (() if method == "native" else ("audit.json",)):
        subprocess.run(["scp", "-q", f"3090-exp:{dest}/{file}", str(local_dir / file)], check=True)
    workload = json.loads((local_dir / "workload.json").read_text())
    assert workload["output_digest"] == EXPECTED_DIGEST, name
    result = {"round": round_id, "method": method,
              "wall_ns": workload["wall_ns"], "output_digest": workload["output_digest"],
              "warmup": 1, "measured": 1, "remote_command": shell,
              "remote_log": f"{dest}/run.log"}
    if method != "native":
        audit = json.loads((local_dir / "audit.json").read_text())
        assert audit["launches"] == 44548 and audit["records"] == 296197072, name
        source = f"{trace}/trace-stream.nvb"
        remote_sha = ssh(f"sha256sum {source}", stdout=subprocess.PIPE, text=True).stdout.split()[0]
        archive = local_dir / "trace-stream.nvb.zst"
        partial = archive.with_suffix(".partial")
        with partial.open("wb") as out:
            with subprocess.Popen(["ssh", "-o", "BatchMode=yes", "3090-exp",
                                   f"{REMOTE}/proton-env/bin/python {REMOTE}/archive_zstd.py {source}"],
                                  stdout=out) as process:
                assert process.wait() == 0
        subprocess.run(["zstd", "-t", str(partial)], check=True, stdout=subprocess.DEVNULL)
        digest = hashlib.sha256()
        with subprocess.Popen(["zstd", "-dc", str(partial)], stdout=subprocess.PIPE) as process:
            while chunk := process.stdout.read(4 << 20):
                digest.update(chunk)
            assert process.wait() == 0
        assert digest.hexdigest() == remote_sha, name
        partial.rename(archive)
        result.update(audit=audit, trace_sha256=remote_sha,
                      local_lossless_archive=str(archive),
                      compressed_bytes=archive.stat().st_size)
        ssh(f"rm -f {source}")
    (local_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    LOCAL.mkdir(exist_ok=True)
    progress = LOCAL / "progress.json"
    rows = json.loads(progress.read_text()) if progress.exists() else []
    rng = random.Random(811044)
    order = list(METHODS)
    rng.shuffle(order)
    for round_id in range(1, 8):
        rotated = order[(round_id - 1) % len(order):] + order[:(round_id - 1) % len(order)]
        for method in rotated:
            existing = LOCAL / f"r{round_id}-{method}" / "result.json"
            if existing.exists():
                continue
            row = remote_run(round_id, method)
            rows.append(row)
            progress.write_text(json.dumps(rows, indent=2) + "\n")
            print(round_id, method, round(row["wall_ns"] / 1e9, 3),
                  row.get("audit", {}).get("container_bytes"), flush=True)
    (LOCAL / "config.json").write_text(json.dumps({"rounds": 7, "seed": 811044,
        "methods": METHODS, "warmup": 1, "measured": 1, "unit": "independent process",
        "endpoint": "request completion plus profile stop and closed trace file",
        "timed_inline_record_validation": False,
        "offline_validation": "every frame structure and byte count; separate full-request same-execution raw oracle",
        "traces_archived_losslessly": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
