import copy,unittest
from neutrino.probe.core_policy import digest_json,resolve_policy,resource_choice,measured_choice

class CorePolicyTests(unittest.TestCase):
    def setUp(self):
        self.r={'shift':dict(registers=40,instructions=200,residency=6,local_bytes=0),
                'late':dict(registers=38,instructions=210,residency=6,local_bytes=0),
                'vector':dict(registers=44,instructions=192,residency=5,local_bytes=0),
                'late_vector':dict(registers=42,instructions=192,residency=6,local_bytes=16)}
    def test_resource_safety_and_tradeoff(self):
        self.assertEqual(resource_choice(self.r),'shift')
        self.assertEqual(resource_choice(self.r,'registers'),'late')
        self.r['late']['residency']=7
        self.assertEqual(resource_choice(self.r),'late')
    def test_calibration_is_paired_and_conservative(self):
        t={'shift':[10]*5,'late':[9]*5,'vector':[8]*5,'late_vector':[7]*5}
        self.assertEqual(measured_choice(t,self.r),'late')
        t['late'][-1]=10.1
        self.assertEqual(measured_choice(t,self.r),'shift')
        t['late']=[9.9]*5
        self.assertEqual(measured_choice(t,self.r),'shift')
        t['late']=[9]
        with self.assertRaises(ValueError):measured_choice(t,self.r)
    def test_policy_identity_and_explicit_fallback(self):
        p=dict(schema=1,default='shift',hardware_digest=digest_json({'arch':'86'}),compiler_sources={'c':'x'},compiler_tools={'p':'y'},functions={'f':dict(mode='late',original_sha256='123')})
        args=({'arch':'86'},{'c':'x'},{'p':'y'})
        self.assertEqual(resolve_policy(p,'f','123',*args)['mode'],'late')
        self.assertEqual(resolve_policy(p,'f','456',*args)['reason'],'changed_binary')
        self.assertEqual(resolve_policy(p,'g','123',*args)['reason'],'unseen_function')
        q=copy.deepcopy(p);q['compiler_sources']['c']='z'
        with self.assertRaises(ValueError):resolve_policy(q,'f','123',*args)
        q=copy.deepcopy(p);q['functions']['f']['mode']='arbitrary'
        with self.assertRaises(ValueError):resolve_policy(q,'f','123',*args)

if __name__=='__main__':unittest.main()
