"""Execute frozen paired schedule with no concurrent GPU treatments."""
import json,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1]
schedule=json.loads((r/'allocation.json').read_text())['blocks']
for b in schedule:
 for method in ['native_before']+b['order']+['native_after']:
  out=r/'capture'/f"r{b['block']}-{method}"
  if (out/'complete.json').exists():continue
  print(json.dumps({'event':'START','block':b['block'],'method':method,'unix_time':time.time()}),flush=True)
  subprocess.run([sys.executable,str(r/'code/run_one.py'),method,str(b['block'])],check=True)
  print(json.dumps({'event':'DONE','block':b['block'],'method':method,'unix_time':time.time()}),flush=True)
print('ALL_COMPLETE',flush=True)
