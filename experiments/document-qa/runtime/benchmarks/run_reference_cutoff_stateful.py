"""Exercise shape/stream changes with the selected cutoff, beyond model digests."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
env=dict(os.environ,NEUTRINO_ARENA_CODEC_MIN_KIB='16')
subprocess.run([sys.executable,str(root/'benchmarks/run_reference_stateful.py')],cwd=root,env=env,check=True)
