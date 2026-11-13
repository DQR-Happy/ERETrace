#!/usr/bin/env python3
"""XTrace PTX producer with a loss-checked append collector for one full request.

The public do_trace runner allocates one slab per kernel definition and overwrites it
on repeated launches. This runner keeps its PTX-site and kernel-swap machinery, but
appends one fixed 24-byte record per warp/site with grid, kernel and warp identity.
It deliberately fails on overflow and validates every launch after the run.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import re
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

from gwatch.cuda.profile import ProfileContext
from gwatch.cuda.binary import PTX
from gwatch.cuda.xtrace._ptx_sites import schedule_sites
from gwatch.cuda.xtrace._runner import find_kernel_ptx

ROOT = Path(__file__).resolve().parent
RECORD_BYTES = 24


def bind(lib, name, argtypes, restype):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def launch_warps(launch):
    blocks = launch.grid_dim_x * launch.grid_dim_y * launch.grid_dim_z
    threads = launch.block_dim_x * launch.block_dim_y * launch.block_dim_z
    return blocks * ((threads + 31) // 32)


def probe_lines(event_type, kernel_id, capacity):
    # One elected lane performs a global atomic reservation. The guard makes
    # every later store conditional, including the out-of-capacity path.
    return [
            "\tmov.u32 %xa_r0, %laneid;",
            "\tsetp.eq.u32 %xa_p0, %xa_r0, 0;",
            "\tld.param.u64 %xa_rd0, [gw_trace_buffer];",
            "\t@%xa_p0 atom.global.add.u64 %xa_rd1, [%xa_rd0], 1;",
            f"\t@%xa_p0 setp.lt.u64 %xa_p0, %xa_rd1, {capacity};",
            f"\tmul.lo.u64 %xa_rd2, %xa_rd1, {RECORD_BYTES};",
            "\tadd.u64 %xa_rd2, %xa_rd2, %xa_rd0;",
            "\tadd.u64 %xa_rd2, %xa_rd2, 16;",
            "\tmov.u64 %xa_rd3, %globaltimer;",
            "\tmov.u64 %xa_rd4, %gridid;",
            "\tmov.u32 %xa_r1, %smid;",
            # CTA linear index for a three-dimensional grid.
            "\tmov.u32 %xa_r2, %ctaid.z;",
            "\tmov.u32 %xa_r3, %nctaid.y;",
            "\tmov.u32 %xa_r7, %ctaid.y;",
            "\tmad.lo.u32 %xa_r2, %xa_r2, %xa_r3, %xa_r7;",
            "\tmov.u32 %xa_r3, %nctaid.x;",
            "\tmov.u32 %xa_r7, %ctaid.x;",
            "\tmad.lo.u32 %xa_r2, %xa_r2, %xa_r3, %xa_r7;",
            # CTA-local linear thread index, then warp index.
            "\tmov.u32 %xa_r3, %tid.z;",
            "\tmov.u32 %xa_r4, %ntid.y;",
            "\tmov.u32 %xa_r7, %tid.y;",
            "\tmad.lo.u32 %xa_r3, %xa_r3, %xa_r4, %xa_r7;",
            "\tmov.u32 %xa_r4, %ntid.x;",
            "\tmov.u32 %xa_r7, %tid.x;",
            "\tmad.lo.u32 %xa_r3, %xa_r3, %xa_r4, %xa_r7;",
            "\tshr.u32 %xa_r3, %xa_r3, 5;",
            "\tmov.u32 %xa_r5, %ntid.z;",
            "\tmul.lo.u32 %xa_r5, %xa_r5, %xa_r4;",
            "\tmov.u32 %xa_r6, %ntid.y;",
            "\tmul.lo.u32 %xa_r5, %xa_r5, %xa_r6;",
            "\tadd.u32 %xa_r5, %xa_r5, 31;",
            "\tshr.u32 %xa_r5, %xa_r5, 5;",
            "\tmad.lo.u32 %xa_r2, %xa_r2, %xa_r5, %xa_r3;",
            "\t@%xa_p0 st.volatile.global.u64 [%xa_rd2], %xa_rd3;",
            "\t@%xa_p0 st.volatile.global.u64 [%xa_rd2+8], %xa_rd4;",
            "\t@%xa_p0 st.volatile.global.u32 [%xa_rd2+16], %xa_r2;",
            "\tand.b32 %xa_r1, %xa_r1, 65535;",
            f"\tor.b32 %xa_r1, %xa_r1, {(event_type << 16) | (kernel_id << 24)};",
            "\t@%xa_p0 st.volatile.global.u32 [%xa_rd2+20], %xa_r1;",
            # Consumption is after whole-device synchronization; no concurrent reader.
            "\tadd.u64 %xa_rd5, %xa_rd0, 8;",
            "\t@%xa_p0 atom.global.add.u64 %xa_rd6, [%xa_rd5], 1;",
    ]


def raw_flash_fallback(binary, prototype, kernel_id, capacity):
    """Patch the same PTX sites when G-Watch's parser cannot materialize a huge kernel."""
    raw = f"/tmp/xtrace-raw-{kernel_id}.ptx"
    binary.dump(raw)
    source = Path(raw).read_text()
    entry = re.search(r"(?m)^\.entry " + re.escape(prototype) + r"\(\n", source)
    if entry is None:
        raise RuntimeError(f"raw PTX entry missing: {prototype}")
    sig_end = source.index("\n)\n", entry.end())
    body_open = source.index("\n{\n", sig_end) + 1
    depth = 0
    body_close = None
    for index in range(body_open, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                body_close = index
                break
    if body_close is None:
        raise RuntimeError("unbalanced PTX entry")
    prefix = source[:entry.start()]
    target = source[entry.start():body_close + 1]
    suffix = source[body_close + 1:]
    if not prototype.startswith("_Z18flash_attn_ext_f16"):
        raise RuntimeError("raw fallback limited to checked FlashAttention PTX layout")
    target = target.replace("\n)\n", ",\n.param .u64 gw_trace_buffer\n)\n", 1)
    declarations = ("\n.reg .b16 %xa_rh<2>;\n.reg .b32 %xa_r<12>;"
                    "\n.reg .b64 %xa_rd<12>;\n.reg .pred %xa_p<2>;")
    target = target.replace("\n{\n", "\n{" + declarations + "\n", 1)
    start_anchor = "mov.u64 %SPL, __local_depot"
    start = target.find(start_anchor)
    if start < 0:
        raise RuntimeError("FlashAttention entry anchor absent")
    target = target[:start] + "\n".join(probe_lines(1, kernel_id, capacity)) + "\n" + target[start:]
    exits = re.findall(r"(?m)^ret;$", target)
    if len(exits) != 1:
        raise RuntimeError(f"FlashAttention exit count changed: {len(exits)}")
    target = target.replace("\nret;", "\n" + "\n".join(probe_lines(2, kernel_id, capacity)) + "\nret;", 1)
    patched = f"/tmp/xtrace-raw-patched-{kernel_id}.ptx"
    Path(patched).write_text(prefix + target + suffix)
    ptx = PTX(); ptx.fill(patched)
    return ptx.jit(), 2


def instrument_ptx(binary, prototype, kernel_id, capacity):
    """Insert entry/exit probes into the G-Watch PTX binary using its PTX editor."""
    ptx = binary.clone()
    ptx.parse()
    try:
        kernel = ptx.get_kerneldef_by_name(prototype)
    except Exception:
        return raw_flash_fallback(binary, prototype, kernel_id, capacity)
    sites = schedule_sites(kernel)
    if not sites or any(site.value_width_bits not in (0, 32) for site in sites):
        raise RuntimeError(f"unsupported PTX site layout: {prototype}")
    ptx.add_argument(kernel_name=prototype, arg_type=".u64", arg_name="gw_trace_buffer",
                     is_ptr=True, space=".global", align=0, array_size=0)
    kernel.refresh()
    for kind, count in ((".b16", 2), (".b32", 12), (".b64", 12), (".pred", 2)):
        suffix = {".b16": "rh", ".b32": "r", ".b64": "rd", ".pred": "p"}[kind]
        ptx.allocate_register(kernel_name=prototype, reg_name=f"%xa_{suffix}",
                              reg_type=kind, num=count)
    kernel.refresh()
    for site in sorted(sites, key=lambda s: s.pc, reverse=True):
        event_type = 1 if site.type_str == "SCOPE_START" else 2
        lines = probe_lines(event_type, kernel_id, capacity)
        for line in reversed(lines):
            ptx.insert_instruction(kernel_name=prototype, pc=site.pc,
                                   ptx_code=line, insert_before=True)
    if os.environ.get("XTRACE_DUMP_PTX"):
        ptx.dump(f"/tmp/xtrace-append-{kernel_id}.ptx")
    return ptx.jit(), len(sites)


def run_request(handle, tokens, prefill, argmax, step, sync):
    assert prefill(handle, tokens, 256) == 256
    generated = [argmax(handle)]
    for index in range(63):
        assert step(handle, generated[-1], 256 + index) == 0
        generated.append(argmax(handle))
    assert sync() == 0
    return hashlib.sha256(json.dumps([generated], separators=(",", ":")).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("rms", "all"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capacity-factor", type=float, default=1.02)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = ProfileContext()
    adapter = C.CDLL(os.environ["QWEN_ADAPTER"])
    create = bind(adapter, "qwen_create", [C.c_char_p, C.c_char_p], C.c_void_p)
    destroy = bind(adapter, "qwen_destroy", [C.c_void_p], None)
    fill = bind(adapter, "qwen_fill_token_ids", [C.c_void_p, C.c_int, C.POINTER(C.c_int)], C.c_int)
    prefill = bind(adapter, "qwen_prefill_ids", [C.c_void_p, C.POINTER(C.c_int), C.c_int], C.c_int)
    argmax = bind(adapter, "qwen_argmax", [C.c_void_p], C.c_int)
    step = bind(adapter, "qwen_step", [C.c_void_p, C.c_int, C.c_int], C.c_int)
    cuda = C.CDLL("libcudart.so.12")
    sync = bind(cuda, "cudaDeviceSynchronize", [], C.c_int)
    malloc = bind(cuda, "cudaMalloc", [C.POINTER(C.c_void_p), C.c_size_t], C.c_int)
    free = bind(cuda, "cudaFree", [C.c_void_p], C.c_int)
    memset = bind(cuda, "cudaMemset", [C.c_void_p, C.c_int, C.c_size_t], C.c_int)
    memcpy = bind(cuda, "cudaMemcpy", [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int], C.c_int)
    meminfo = bind(cuda, "cudaMemGetInfo", [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)], C.c_int)
    handle = create(os.environ["QWEN_MODEL"].encode(), os.environ["QWEN_BACKEND"].encode())
    if not handle:
        raise RuntimeError("model/context creation failed")
    device_ptr = C.c_void_p()
    registered = []
    try:
        tokens = (C.c_int * 256)()
        assert fill(handle, 256, tokens) == 256
        ctx.start_tracing_kernel_launch()
        warm_digest = run_request(handle, tokens, prefill, argmax, step, sync)
        launches = ctx.stop_tracing_kernel_launch()
        chosen = [x for x in launches if args.scope == "all" or "rms_norm" in x.definition.mangled_prototype.lower()]
        names = sorted({x.definition.mangled_prototype for x in chosen})
        name_id = {name: i + 1 for i, name in enumerate(names)}
        expected_warps = sum(launch_warps(x) for x in chosen)
        capacity = int(expected_warps * 2 * args.capacity_factor) + 1024
        available = C.c_size_t(); total = C.c_size_t()
        assert meminfo(C.byref(available), C.byref(total)) == 0
        allocation_bytes = 16 + capacity * RECORD_BYTES
        print(json.dumps({"phase": "scout", "scope": args.scope, "launches": len(chosen),
                          "kernels": len(names), "expected_warps": expected_warps,
                          "capacity_bytes": allocation_bytes, "free_bytes": available.value,
                          "warm_digest": warm_digest}), flush=True)
        if allocation_bytes > available.value - 1024**3:
            raise MemoryError("append buffer would leave <1 GiB GPU headroom")
        assert malloc(C.byref(device_ptr), allocation_bytes) == 0
        assert memset(device_ptr, 0, 16) == 0
        binaries = ctx.get_list_binary_ptx()
        compilation = []
        for name in names:
            binary = find_kernel_ptx(name, "", binaries)
            if binary is None:
                raise RuntimeError(f"PTX not available: {name}")
            image, site_count = instrument_ptx(binary, name, name_id[name], capacity)
            ctx.load_binary_cubin(binary_cubin=image)
            replacement = ctx.load_kerneldef_sass(binary_cubin=image, list_load_kernel_name=[name])[0]
            original = ctx.get_kernel_def_sass_by_name(name)
            if original is None:
                raise RuntimeError(f"original SASS definition missing: {name}")
            ctx.register_trace_kernel_sass(original, replacement, param_values=[device_ptr.value])
            registered.append(original)
            compilation.append({"kernel_id": name_id[name], "name": name, "sites": site_count})
            print(json.dumps({"phase": "compiled", "count": len(compilation),
                              "total": len(names), "name": name}), flush=True)
        ctx.start_tracing_kernel_launch()
        begin = time.perf_counter_ns()
        traced_digest = run_request(handle, tokens, prefill, argmax, step, sync)
        trace_ns = time.perf_counter_ns() - begin
        traced_launches = ctx.stop_tracing_kernel_launch()
        for original in reversed(registered):
            ctx.unregister_trace_kernel_sass(original)
        registered.clear()
        counter = C.c_uint64()
        assert memcpy(C.byref(counter), device_ptr, 8, 2) == 0
        if counter.value > capacity:
            raise RuntimeError(f"TRACE OVERFLOW: {counter.value} > {capacity}")
        chosen_traced = [x for x in traced_launches if x.definition.mangled_prototype in name_id]
        expected_traced = 2 * sum(launch_warps(x) for x in chosen_traced)
        print(json.dumps({"phase": "captured", "trace_ns": trace_ns,
                          "records": counter.value, "expected_records": expected_traced,
                          "digest": traced_digest}), flush=True)
        if counter.value != expected_traced or traced_digest != warm_digest:
            raise RuntimeError("record count or model output differs from scout request")
        path = out.with_suffix(".trace")
        chunk = C.create_string_buffer(64 * 1024 * 1024)
        copied = 0
        h = hashlib.sha256()
        copy_begin = time.perf_counter_ns()
        with open(path, "wb", buffering=0) as stream:
            while copied < counter.value * RECORD_BYTES:
                n = min(len(chunk), counter.value * RECORD_BYTES - copied)
                assert memcpy(chunk, C.c_void_p(device_ptr.value + 16 + copied), n, 2) == 0
                block = memoryview(chunk).cast("B")[:n]
                stream.write(block)
                h.update(block)
                copied += n
        copy_ns = time.perf_counter_ns() - copy_begin
        result = {"scope": args.scope, "scout_launches": len(chosen),
                  "traced_launches": len(chosen_traced), "kernels": compilation,
                  "expected_warps": expected_warps, "record_count": counter.value,
                  "trace_ns": trace_ns, "trace_ms": trace_ns / 1e6,
                  "copy_and_write_ns": copy_ns, "completion_ms": (trace_ns + copy_ns) / 1e6,
                  "trace_bytes": copied, "trace_sha256": h.hexdigest(),
                  "trace_path": str(path), "output_digest": traced_digest,
                  "record_layout": "<QQIHBB: timestamp_ns, gridid, warp_id, smid, type, kernel_id",
                  "note": "modified XTrace PTX append collector; JIT/scout excluded; launch tracing included"}
        out.with_suffix(".json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"phase": "done", **{k: result[k] for k in
                          ("trace_ms", "completion_ms", "trace_bytes", "record_count", "trace_sha256")}}), flush=True)
    finally:
        for original in reversed(registered):
            ctx.unregister_trace_kernel_sass(original)
        if device_ptr.value:
            free(device_ptr)
        destroy(handle)


if __name__ == "__main__":
    main()
