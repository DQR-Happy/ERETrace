import unittest
from neutrino.common import Map, Probe, Register
from neutrino.probe.cuda import probing, _instruction_matches


def fixtures(body="ret;", params=".param .u64 data"):
    asm = f".version 8.4\n.target sm_86\n.address_size 64\n.visible .entry test(\n{params}\n)\n{{\n.reg .pred %p;\n.reg .b64 %rd;\n{body}\n}}"
    maps = [Map("Trace", "warp", "array", 16, 1,
                [Register("start", "u64"), Register("duration", "u32"), Register("sm", "u32")])]
    probes = [Probe("sched", "warp", ["kernel"], "mov.u64 %NR0, %clock64;",
                    "mov.u64 %NR1, %clock64;\nsub.u64 %NR1, %NR1, %NR0;\nmov.u32 %Trace_u32_1, %smid;\ncvt.u64.u32 %NR2, %Trace_u32_1;\nSAVE [Trace] {%NR0, %NR1, %NR2};")]
    return asm, probes, maps, 3


class CoreTests(unittest.TestCase):
    def test_opcode_matching(self):
        for line in ("// ret;", ".reg .u32 %ret;", "mov.u32 %ret, 1;", "Lret:"):
            self.assertFalse(_instruction_matches(line, "ret;"))
        self.assertTrue(_instruction_matches("@!%p ret; // comment", "ret;"))
        self.assertTrue(_instruction_matches("ld.global.u32 %r, [%rd];", "ld.global"))

    def test_partial_warp_stride(self):
        for n in range(1, 1025):
            stride = (n + 31) // 32
            ids = [cta * stride + lane // 32 for cta in range(5) for lane in range(0, n, 32)]
            self.assertEqual(ids, list(range(5 * stride)))
        self.assertIn("add.u32 %warpbuf20, %warpbuf20, 31;", probing(*fixtures()))

    def test_positive_negative_exit(self):
        code = probing(*fixtures("@%p ret;\n@!%p exit;"))
        self.assertIn("and.pred %joint_pred, %leader, %p;", code)
        self.assertIn("not.pred %joint_pred, %p;", code)
        self.assertNotIn("%tmp", code)
        self.assertEqual(code.count("st.global.u64"), 2)

    def test_late_and_vector(self):
        code = probing(*fixtures("@%p ret;\nexit;"), core_mode="late_vector")
        self.assertEqual(code.count("st.global.v4.u32"), 2)
        self.assertEqual(code.count("// begin Trace map"), 2)
        self.assertGreater(code.index("// begin Trace map"), code.index("sub.u64 %NR1"))
        self.assertNotIn("div.s32 %warpbuf", code)
        scalar = probing(*fixtures(), core_mode="late")
        self.assertEqual(scalar.count("// begin Trace map"), 1)

    def test_late_rejects_cursors(self):
        asm, probes, maps, regs = fixtures()
        maps[0].cap = 4
        with self.assertRaises(ValueError):
            probing(asm, probes, maps, regs, "late")

    def test_empty_parameter_list(self):
        code = probing(*fixtures(params=""))
        self.assertIn(".param .u64 param_Trace", code)
        self.assertNotIn("(,", code)

    def test_address_offset(self):
        asm, probes, maps, regs = fixtures("ld.global.u64 %rd, [%rd+16];\nret;")
        probes = [Probe("mem", "warp", ["ld.global"], "mov.u64 %NR0, ADDR;", None)]
        code = probing(asm, probes, maps, regs)
        self.assertIn("add.u64 %core_addr, %rd, 16;", code)


if __name__ == "__main__":
    unittest.main()
