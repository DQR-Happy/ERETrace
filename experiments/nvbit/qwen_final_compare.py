#!/usr/bin/env python3
"""Final paired tuned ISPA versus optimized Raw and nvCOMP Cascaded."""
import json
from pathlib import Path
import random
import shlex
import subprocess

import qwen_matrix_fair as base

LOCAL = base.HERE / "qwen-matrix-final"
METHODS = ("hybrid", "raw", "nvcomp")
SO = f"{base.REMOTE}/qwen_scope_async/qwen_scope_async.so"


def run(round_id, method):
    name = f"r{round_id}-{method}"
    dest = f"{base.REMOTE}/qwen-matrix-final/{name}"
    trace = f"/dev/shm/review1-qwen-final-{name}"
    base.ssh(f"mkdir -p {shlex.quote(dest)} {shlex.quote(trace)}")
    env = {"PATH": "/usr/local/cuda/bin:$PATH",
           "LD_LIBRARY_PATH": f"{base.BACKEND}:${{LD_LIBRARY_PATH:-}}",
           "NEUTRINO_BENCH_DIR": dest,
           "REVIEW_NVBIT_PROFILE": "1", "REVIEW_NVBIT_WARMUP_PROFILE": "1",
           "NVBIT_CAPTURE_ALL": "1", "NVBIT_STREAM": "1", "NVBIT_QUIET": "1",
           "ACTIVE_FROM_START": "0", "NVBIT_MAX_WARPS": "607744",
           "NVBIT_ASYNC_SLOTS": "4", "SCOPE_TRACE_DIR": trace,
           "CUDA_INJECTION64_PATH": SO, "NVBIT_BACKEND": method}
    if method == "hybrid":
        env.update(NVBIT_LOCAL_BITS="16", NVBIT_ELAPSED_BYTES="3",
                   NVBIT_ENCODER_CUBIN=base.ENCODER)
    elif method == "nvcomp":
        env["NVBIT_NVCOMP_LIB"] = f"{base.REMOTE}/nvcomp_pipeline.so"
    envs = " ".join(f"{k}={v if '$' in v else shlex.quote(v)}" for k, v in env.items())
    command = ["python3", "qwen_exact_nvbit.py", "--adapter", base.ADAPTER,
               "--backend", base.BACKEND, "--model", base.MODEL,
               "--warmup", "1", "--requests", "1", "--input-tokens", "256",
               "--output-tokens", "64", "--batch-size", "1"]
    shell = (f"cd {base.REMOTE} && env {envs} timeout 600 "
             + " ".join(shlex.quote(x) for x in command)
             + f" >{dest}/run.log 2>&1")
    base.ssh(shell)
    base.ssh(f"python3 {base.REMOTE}/qwen_stream.py {trace}/trace-stream.nvb >{dest}/audit.json")
    local = LOCAL / name
    local.mkdir(parents=True, exist_ok=True)
    for filename in ("run.log", "workload.json", "audit.json"):
        subprocess.run(["scp", "-q", f"3090-exp:{dest}/{filename}", str(local / filename)], check=True)
    workload = json.loads((local / "workload.json").read_text())
    audit = json.loads((local / "audit.json").read_text())
    assert workload["output_digest"] == base.EXPECTED_DIGEST
    assert audit["launches"] == 44548 and audit["records"] == 296197072
    source = f"{trace}/trace-stream.nvb"
    remote_sha = base.ssh(f"sha256sum {source}", stdout=subprocess.PIPE, text=True).stdout.split()[0]
    result = {"round": round_id, "method": method, "wall_ns": workload["wall_ns"],
              "output_digest": workload["output_digest"], "audit": audit,
              "trace_sha256": remote_sha, "remote_command": shell}
    result["trace_retention"] = "structurally audited and SHA256 recorded; representative full streams retained in qwen-matrix-async and qwen-matrix-hybrid"
    base.ssh(f"rm -f {source}")
    (local / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    LOCAL.mkdir(exist_ok=True)
    rng = random.Random(811048)
    for round_id in range(1, 8):
        methods = list(METHODS)
        rng.shuffle(methods)
        for method in methods:
            existing = LOCAL / f"r{round_id}-{method}" / "result.json"
            if existing.exists():
                continue
            row = run(round_id, method)
            print(round_id, method, round(row["wall_ns"] / 1e9, 3),
                  row["audit"]["container_bytes"], flush=True)
    (LOCAL / "config.json").write_text(json.dumps({"rounds": 7, "seed": 811048,
        "methods": METHODS, "max_warps": 607744, "slots": 4,
        "comparison": "same SASS probe, same 16B records and bounded async collector",
        "unit": "independent process, one instrumented warmup and one measured request",
        "endpoint": "profiler stop, async queue drained and complete file closed"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
