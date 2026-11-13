import unittest
from neutrino.probe.core_capacity import capacity_classes,event_bound,private_layout,instrument_stratified_events
from test_core_events import PTX

class CapacityTests(unittest.TestCase):
    def test_periodic_upper_bounds(self):
        n=('seq',('events',2),('residue',4,[1],('repeat',4,('events',2)),('events',0)))
        self.assertEqual(capacity_classes(n),[2,10,2,2])
        self.assertEqual(event_bound(('unknown',n,('events',20)),1),20)
    def test_prefix_partition_with_partial_cta(self):
        for caps in ([0,3],[2,9,2,18],[1]):
            for t in (1,31,32,33,48,63,127,1024):
                rows,total=private_layout(caps,t,3)
                slots=[i+j for i,n in rows for j in range(n)]
                self.assertEqual(slots,list(range(total)))
                self.assertEqual(total,sum(caps[i%len(caps)] for i in range(t))*3)
    def test_emit_two_equivalent_layouts(self):
        for mode in ['periodic','table']:
            text,sites=instrument_stratified_events(PTX,[2,9,2,18],mode)
            self.assertEqual(len(sites),2)
            self.assertIn('setp.lt.u64 %obs_space, %obs_index, %obs_capacity;',text)
            self.assertIn('shl.b64 %obs_off, %obs_slot, 5;',text)
        self.assertIn('obs_class_prefix[5]',instrument_stratified_events(PTX,[2,9,2,18])[0])
    def test_reject_bad_bounds(self):
        for caps in ([0],[1,2,3],[-1,3],[65537]):
            with self.assertRaises(ValueError):private_layout(caps,32,1)
        with self.assertRaises(ValueError):event_bound(('repeat',-1,('events',1)),0)
        with self.assertRaises(ValueError):capacity_classes(('residue',3,[0],('events',1),('events',2)))

if __name__=='__main__':unittest.main()
