"""Callback fixture: validate a Neutrino trace without loading a GPU library."""
from pathlib import Path
import struct
import sys

path = Path(sys.argv[1])
with path.open('rb') as file:
    header = struct.unpack('<8I', file.read(32))
assert header[-1] >= 1 and path.stat().st_size > 48
