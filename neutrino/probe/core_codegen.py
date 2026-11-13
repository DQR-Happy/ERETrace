"""Conservative, opt-in fixed-record code-generation experiments.

These passes change address lifetime and store width, not sampling sites or
record fields. The late-address transform deliberately rejects general probes:
a cursor that advances in a loop cannot be reconstructed from a warp index.
"""
import re

MODES = {"corrected", "shift", "late", "vector", "late_vector"}


def validate_mode(mode, probes, maps):
    if mode not in MODES:
        raise ValueError(f"Unknown NEUTRINO_CORE_MODE: {mode}")
    if mode in {"late", "late_vector"}:
        valid = (len(maps) == 1 and maps[0].level == "warp"
                 and maps[0].cap == 1 and maps[0].size == 16
                 and [r.dtype for r in maps[0].regs] == ["u64", "u32", "u32"]
                 and len(probes) == 1 and probes[0].level == "warp"
                 and probes[0].pos == ["kernel"]
                 and probes[0].before is not None and "SAVE" not in probes[0].before
                 and probes[0].after is not None and probes[0].after.count("SAVE") == 1)
        if not valid:
            raise ValueError("late address mode requires one fixed warp kernel-entry/exit record")


def finish_codegen(ptx, mode, maps):
    if mode in {"shift", "late", "late_vector"}:
        # CUDA block dimensions are positive and block threads <= 1024.
        # Keep the separate 32-bit global index limit explicit in the docs.
        ptx = ptx.replace("div.s32 %warpbuf15, %warpbuf15, 32;", "shr.u32 %warpbuf15, %warpbuf15, 5;")
        ptx = ptx.replace("div.s32 %warpbuf20, %warpbuf20, 32;", "shr.u32 %warpbuf20, %warpbuf20, 5;")
        ptx = re.sub(r"^@%leader mov.u32 %warpbuf3, %nwarpid;.*\n", "", ptx, flags=re.M)
    if mode in {"vector", "late_vector"}:
        for m in maps:
            if m.size != 16 or m.cap == "dynamic" or [r.dtype for r in m.regs] != ["u64", "u32", "u32"]:
                continue
            # CUDA allocation is aligned; this fixed map has 16-byte stride.
            n = re.escape(m.name)
            pattern = (rf"(?P<p>@[!]?%[\w$]+ )?st.global.u64 \[%map_{n}1\], (?P<x>[^;]+);\s*"
                       rf"(?P=p)add.s64 %map_{n}1, %map_{n}1, 8;\s*"
                       rf"(?P=p)st.global.v2.u32 \[%map_{n}1\],\s*\{{\s*(?P<a>[^,]+),\s*(?P<b>[^}}]+)\s*\}};\s*"
                       rf"(?P=p)add.s64 %map_{n}1, %map_{n}1, 8;")
            def pack(match):
                p = match['p'] or ''
                return (".reg .b32 %core_low, %core_high;\n"
                        f"{p}mov.b64 {{%core_low, %core_high}}, {match['x']};\n"
                        f"{p}st.global.v4.u32 [%map_{m.name}1], {{%core_low, %core_high, {match['a']}, {match['b']}}};\n"
                        f"{p}add.s64 %map_{m.name}1, %map_{m.name}1, 16;")
            ptx, count = re.subn(pattern, pack, ptx)
            if not count:
                raise ValueError("vector-store transform found no supported SAVE expansion")
    if mode in {"late", "late_vector"}:
        m = maps[0]
        calc = re.search(r"// begin buffer calculation\n.reg .b32   %warpbuf.*?// end buffer calculation", ptx, re.S)
        map_calc = re.search(rf"// begin {re.escape(m.name)} map\n.*?// end {re.escape(m.name)} map", ptx, re.S)
        if calc is None or map_calc is None:
            raise ValueError("late address transform cannot locate its fixed templates")
        old = calc.group()
        local = '\n'.join(line for line in old.splitlines()
                          if not (".reg .pred" in line or "%warpbuf2, %laneid" in line
                                  or "setp.eq.u32 %leader" in line))
        leader = (".reg .pred %leader, %joint_pred;\n.reg .b32 %core_lane;\n"
                  "mov.u32 %core_lane, %laneid;\nsetp.eq.u32 %leader, %core_lane, 0;")
        ptx = ptx.replace(old, leader, 1).replace(map_calc.group(), '', 1)
        # The surrounding Ref scope is unique for each exit, so temporary
        # declarations do not collide. Clock/SM snapshots precede this point.
        ptx = re.sub(rf"(?m)^(?P<line>@[!]?%[\w$]+ st.global\.[^\n]*\[%map_{re.escape(m.name)}1\][^\n]*)",
                     lambda x: local + '\n' + map_calc.group() + '\n' + x['line'], ptx, count=0)
        # For scalar SAVE there are two stores, but only the first must reset
        # the cursor. Repeated setup would overwrite the first eight bytes.
        # Remove setup before a subsequent store in the same Ref scope.
        duplicate = re.escape(local + '\n' + map_calc.group() + '\n')
        ptx = re.sub(rf"(add.s64 %map_{re.escape(m.name)}1, %map_{re.escape(m.name)}1, 8;\s*){duplicate}", r"\1", ptx)
    return ptx
