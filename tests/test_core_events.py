import unittest
from neutrino.probe.core_events import instrument_memory_events

PTX = '''.version 8.4
.target sm_86
.address_size 64
.visible .entry f(
.param .u64 p
)
{
.reg .pred %a;
.reg .u64 %ptr;
.reg .u32 %v;
// ld.global.u32 %v, [%ptr];
@!%a ld.global.u32 %v, [%ptr-16];
st.global.u32 [%ptr+4], %v;
@%a ret;
exit;
}'''

class CoreEventTests(unittest.TestCase):
    def test_sites_predicate_and_address(self):
        text, sites = instrument_memory_events(PTX, 32)
        self.assertEqual(len(sites), 2)
        self.assertIn('not.pred %obs_enabled, %a;', text)
        self.assertIn('sub.u64 %obs_addr, %ptr, 16;', text)
        self.assertIn('add.u64 %obs_addr, %ptr, 4;', text)
        self.assertEqual(text.count('st.global.u64 [%obs_count_base]'), 2)
        self.assertIn('setp.lt.u64 %obs_space, %obs_index, 32;', text)

    def test_private_slot_injectivity(self):
        for cap in (1, 4, 32):
            slots = [t*cap+j for t in range(257) for j in range(cap)]
            self.assertEqual(slots, list(range(257*cap)))

    def test_rejects_unhandled_inputs(self):
        for cap in (0, 65537):
            with self.assertRaises(ValueError):instrument_memory_events(PTX,cap)
        with self.assertRaises(ValueError):
            instrument_memory_events(PTX.replace('%ptr-16','%ptr+%v'),32)
        with self.assertRaises(ValueError):instrument_memory_events(PTX+PTX,32)

    def test_rejects_collisions_and_hidden_sites(self):
        for old,new in [('%v','%obs_count'),('%v','%buf1'),
                        ('st.global.u32','LABEL: st.global.u32'),
                        ('[%ptr+4], %v;','[%ptr+4], %v; add.u32 %v, %v, 1;'),
                        ('exit;','DONE: exit;')]:
            with self.assertRaises(ValueError):instrument_memory_events(PTX.replace(old,new),32)

if __name__ == '__main__':unittest.main()
