"""CPU tests for accounting, concurrency, sampling, snapshots and fork isolation."""
import json
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from neutrino.timing_report import summarize

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = r'''
#include "timing.h"
#include <sys/wait.h>
static void *worker(void *arg) {
    for (int i = 0; i < 64; ++i) {
        nt_span timing = nt_begin((void *)(uintptr_t)(16 + i % 2));
        timing.name = "a\"b\\c\n";
        timing.cold = i == 0;
        NT_STAGE(HOST_ALLOC);
        NT_STAGE(DEVICE_ALLOC);
        NT_STAGE(HOST_ALLOC); /* repeated stage counts once per invocation */
        timing.gpu_valid = 1; timing.gpu_ms = 0.001;
        timing.trace_bytes = 128; timing.d2h_bytes = 80;
        nt_finish(&timing, NT_CAPTURED, 0);
    }
    return arg;
}
int main(int argc, char **argv) {
    nt_init(argv[1]);
    pthread_t threads[4];
    for (int i = 0; i < 4; ++i) pthread_create(&threads[i], NULL, worker, NULL);
    for (int i = 0; i < 4; ++i) pthread_join(threads[i], NULL);
    if (argc > 2 && strcmp(argv[2], "overflow") == 0) {
        for (int i = 0; i < 1200; ++i) {
            nt_span timing = nt_begin((void *)(uintptr_t)(0x10000 + i * 16));
            nt_finish(&timing, NT_FALLBACK, 1);
        }
    }
    if (argc > 2 && strcmp(argv[2], "reset") == 0) {
        if (neutrino_timing_flush() != 0 || neutrino_timing_reset() != 0) return 2;
        worker(NULL);
    }
    /* The child must not publish an inherited parent aggregate, even via exit(). */
    pid_t child = fork();
    if (child == 0) { worker(NULL); exit(0); }
    waitpid(child, NULL, 0);
    return 0;
}
'''


class HostTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source = cls.root / 'fixture.c'
        source.write_text(FIXTURE)
        cls.exe = cls.root / 'fixture'
        subprocess.run([shutil.which('cc'), '-std=gnu11', '-O2', '-pthread',
                        '-I', str(ROOT / 'neutrino/src'), str(source), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_fixture(self, sample=1, mode='normal', enabled=True):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, NEUTRINO_TIMING=str(int(enabled)),
                       NEUTRINO_TIMING_DIR=directory,
                       NEUTRINO_TIMING_SAMPLE_EVERY=str(sample), NEUTRINO_TIMING_FLUSH_EVERY='80')
            subprocess.run([str(self.exe), directory, mode], env=env, check=True)
            files = list(Path(directory).glob('timing-*.json'))
            self.assertEqual(len(files), int(enabled))
            self.assertFalse(list(Path(directory).glob('*.tmp')))
            if not enabled:
                return None
            result = json.loads(files[0].read_text())
            for group in result['groups']:
                self.assertEqual(group['host_wall']['total_ns'],
                                 sum(v['total_ns'] for v in group['stages'].values()))
                self.assertEqual(group['host_wall']['samples'], group['sampled'])
                for stat in group['stages'].values():
                    self.assertGreaterEqual(stat['total_ns'], stat['min_ns'] * stat['samples'])
                    self.assertLessEqual(stat['total_ns'], stat['max_ns'] * stat['samples'])
            self.assertEqual(sum(g['calls'] for g in result['groups']), result['calls_completed'])
            return result

    def test_partition_concurrency_and_fork(self):
        result = self.run_fixture()
        self.assertEqual(result['calls_completed'], 256)
        self.assertEqual(result['calls_sampled'], 256)
        self.assertEqual(sum(g['sampled_trace_bytes'] for g in result['groups']), 256 * 128)
        self.assertEqual(sum(g['gpu_probe_interval']['total_ns'] for g in result['groups']), 256 * 1000)
        self.assertTrue(all(g['kernel'] == 'a"b\\c\n' for g in result['groups']))
        self.assertEqual(result['snapshot_kind'], 'normal_exit')
        self.assertEqual(result['previous_snapshots'], 3)

    def test_sampling_keeps_exact_call_counts(self):
        result = self.run_fixture(sample=7)
        self.assertEqual(result['calls_completed'], 256)
        self.assertEqual(result['calls_sampled'], 40)  # 10 per host thread

    def test_reset_window(self):
        result = self.run_fixture(mode='reset')
        self.assertEqual(result['calls_completed'], 64)
        self.assertEqual(result['calls_sampled'], 64)

    def test_bounded_groups_preserve_totals(self):
        result = self.run_fixture(mode='overflow')
        self.assertEqual(result['calls_completed'], 1456)
        self.assertEqual(len(result['groups']), 1024)
        self.assertTrue(any(g['outcome'] == 'overflow' for g in result['groups']))
        self.assertEqual(sum(g['errors'] for g in result['groups']), 1200)

    def test_disabled_writes_nothing(self):
        self.run_fixture(enabled=False)

    def test_report_rejects_corrupt_budgets(self):
        result = self.run_fixture()
        self.assertEqual(sum(r['calls'] for r in summarize([result]).values()), 256)
        bad = copy.deepcopy(result)
        bad['groups'][0]['host_wall']['total_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'partition'):
            summarize([bad])
        bad = copy.deepcopy(result)
        bad['calls_sampled'] -= 1
        with self.assertRaisesRegex(ValueError, 'Sampled'):
            summarize([bad])


if __name__ == '__main__':
    unittest.main()
