"""Run CPU output tests after GPU measurements to avoid shared-host contention."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
root=Path(__file__).resolve().parents[1];out=root/'results/deep'
while not (out/'stream-complete.json').exists():
    pid=json.loads((out/'stream-pid.json').read_text())['pid']
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError('Stream experiment did not complete')
    time.sleep(2)
temporary=Path('/dev/shm/neutrino-file-deep');temporary.mkdir(exist_ok=False)
try:
    with (out/'file-micro.json').open('w') as fp:
        subprocess.run([str(root/'benchmarks/file_path_micro'),str(root/'benchmarks/file-frame-spec.txt'),str(temporary),'7'],stdout=fp,check=True,timeout=300)
    d=json.loads((out/'file-micro.json').read_text());assert d['size_validation'] and d['bytes_per_run']==652849408 and d['frames_per_run']==16648
    (out/'file-micro-complete.json').write_text(json.dumps({'complete':True})+'\n')
finally:shutil.rmtree(temporary)
