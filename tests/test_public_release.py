"""Check public portability wrappers against the full capture contract."""
import json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PublicReleaseTest(unittest.TestCase):
    def test_b16_selects_extended_adapter_and_workload(self):
        cmd=[sys.executable,str(ROOT/'scripts/reproduce.py'),'--dry-run','--model','/tmp/model.gguf','--backend','/tmp/backend','--output','/tmp/not-created','--batch-size','16','--context','8192']
        result=json.loads(subprocess.check_output(cmd,text=True));args=result['command']
        self.assertIn(str(ROOT/'experiments/b16/workload.py'),args)
        self.assertIn(str(ROOT/'build/libllama_adapter_b16.so'),args)
        self.assertEqual(result['specs']['ours']['env']['NEUTRINO_BENCH_SEQUENCES'],'16')
    def test_case_raw_and_ere_have_equal_arena_budget(self):
        specs={}
        for method in ['async','ours']:
            cmd=[sys.executable,str(ROOT/'scripts/case.py'),'run','--dry-run','--condition',method,'--model','/tmp/model.gguf','--backend','/tmp/backend','--output','/tmp/not-created']
            specs[method]=json.loads(subprocess.check_output(cmd,text=True))['environment']
        for key in ['NEUTRINO_ARENA_HOST_MIB','NEUTRINO_ARENA_DEVICE_MIB','NEUTRINO_ARENA_CHUNK_MIB']:
            self.assertEqual(specs['async'][key],specs['ours'][key])
        self.assertEqual(specs['async']['NEUTRINO_CASE_RAW'],'1')
        self.assertEqual(specs['ours']['CASE_ANALYZE'],'0')
        self.assertEqual(specs['ours']['NEUTRINO_ARENA_REFERENCE'],'1')
    def test_case_build_targets_isolated_case_runtime(self):
        cmd=[sys.executable,str(ROOT/'scripts/case.py'),'build','--dry-run','--llama-source','/tmp/llama','--llama-build','/tmp/llama-build']
        commands=json.loads(subprocess.check_output(cmd,text=True))
        self.assertIn(str(ROOT/'experiments/document-qa/runtime'),commands[0])
        self.assertIn(str(ROOT/'experiments/document-qa/runtime/benchmarks/llama_adapter_dynamic.cpp'),commands[1])
if __name__=='__main__':unittest.main()
