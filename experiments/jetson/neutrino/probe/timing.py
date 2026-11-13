"""Cold-path timing; the parent measures the complete fork/exec/wait interval."""
import json
import os
import time


class PhaseTimer:
    def __init__(self, workdir, entry_ns):
        self.enabled = os.environ.get('NEUTRINO_TIMING') == '1'
        self.path = os.path.join(workdir, 'process-timing.json')
        self.start = self.last = entry_ns
        self.phase = 'python_imports'
        self.stages = {}
        self.status = 'incomplete'

    def stage(self, phase):
        if not self.enabled:
            return
        now = time.perf_counter_ns()
        self.stages[self.phase] = self.stages.get(self.phase, 0) + now - self.last
        self.last, self.phase = now, phase

    def finish(self):
        if not self.enabled:
            return
        self.stage('report_write')
        payload = dict(schema_version=1, pid=os.getpid(), status=self.status,
                       clock='perf_counter_ns', total_ns=self.last - self.start,
                       stages_ns=self.stages)
        try:
            with open(self.path + '.tmp', 'w') as f:
                json.dump(payload, f, indent=2)
                f.write('\n')
            os.replace(self.path + '.tmp', self.path)
        except OSError as exc:
            # Instrumentation output must not turn a successful compilation into fallback.
            import sys
            print(f'Neutrino probe timing: {exc}', file=sys.stderr)
