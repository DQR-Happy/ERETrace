import importlib.util,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('summary',ROOT/'scripts/summarize_runs.py');summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
class ArtifactToolsTest(unittest.TestCase):
    def test_paired_rounds_and_incomplete_input(self):
        rows=[{'condition':c,'round':r,'requests':[{}]*3,'wall_ns':value*3*1e6} for r in range(1,4) for c,value in [('native',r+80),('ours',r+100)]]
        result=summary.summarize(rows);self.assertEqual(result[1]['delta_native_ms'],20)
        self.assertEqual(result[1]['paired_mean_ci95_low_ms'],20)
        with self.assertRaises(ValueError):summary.summarize(rows[:-1])
        with self.assertRaises(ValueError):summary.summarize(rows+[rows[0]])
    def test_dry_run_uses_matched_pipeline(self):
        s=subprocess.check_output([sys.executable,str(ROOT/'scripts/reproduce.py'),'--dry-run','--model','/tmp/model.gguf','--backend','/tmp/backend','--output','/tmp/test-output','--conditions','ours','sm24','nvcomp'],text=True)
        d=json.loads(s)['specs'];self.assertEqual(d['ours']['env']['NEUTRINO_ARENA_DEVICE_MIB'],d['sm24']['env']['NEUTRINO_ARENA_DEVICE_MIB'])
        self.assertEqual(d['sm24']['env']['NEUTRINO_ARENA_REFERENCE'],'0')
        self.assertEqual(d['nvcomp']['env']['NEUTRINO_ARENA_NVCOMP'],'2')
    def test_recorded_recovery_accounting(self):
        data=json.loads((ROOT/'results/paper/main-recovery.json').read_text())
        for r in data['summaries']:
            t=r['totals'];self.assertEqual(t['all_maps'],t['encoded_maps']+t['direct_raw_maps'])
            self.assertAlmostEqual(r['extra_raw_D2H_B_per_request']*r['requests'],t['extra_raw_D2H_B'])
if __name__=='__main__':unittest.main()
