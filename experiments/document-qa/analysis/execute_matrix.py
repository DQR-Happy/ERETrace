"""Local controller: serial GPU work, full formal archives split across hosts."""
from pathlib import Path
import subprocess,json,shutil,hashlib,shlex,time
r=Path(__file__).resolve().parents[1];remote='/root/autodl-tmp/neutrino-inference-case-20261013'
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','3090-new']
def sync_metadata(name,out):
 cmd=['rsync','-az','--partial','--exclude','*.zst','-e',shlex.join(ssh[:-1]),f'3090-new:{remote}/capture/{name}/',str(out)+'/']
 for attempt in range(4):
  if subprocess.call(cmd)==0:return
  time.sleep(min(20,5*(attempt+1)))
 raise RuntimeError(f'{name}: metadata transfer failed; GPU data retained remotely')
allocation=json.loads((r/'allocation.json').read_text());records=[]
for block in allocation['rounds']:
 for condition in block['capture_order']:
  while (r/'pause-matrix').exists():time.sleep(1)
  name=f'r{block["round"]}-{condition}';out=r/'capture'/name;out.mkdir(parents=True,exist_ok=True)
  if not (out/'complete.json').exists():
   print('START',name,flush=True)
   cmd=ssh+[f'/root/miniconda3/bin/python {remote}/code/run_one.py {condition} {block["round"]}']
   with (out/'controller.log').open('w') as log:
    rc=subprocess.call(cmd,stdout=log,stderr=log)
   if rc:raise RuntimeError(f'{name} failed: {rc}; inspect controller.log and remote log')
   sync_metadata(name,out)
  archives=json.loads((out/'archives.json').read_text());places=[]
  for entry in archives:
   basename=Path(entry['archive']).name;dest=out/basename
   if dest.exists():
    h=hashlib.sha256()
    with dest.open('rb') as f:
     for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    assert h.hexdigest()==entry['archive_sha256']
    code='from pathlib import Path;Path('+repr(entry['archive'])+').unlink(missing_ok=True)'
    subprocess.run(ssh+['/root/miniconda3/bin/python -c '+shlex.quote(code)],check=True)
    places.append({**entry,'retained_host':'local','retained_path':str(dest),'transfer_sha256_verified':True})
   else:places.append({**entry,'retained_host':'3090-new','retained_path':entry['archive'],'transfer_policy':'retain remotely; metadata copied locally; full archive already roundtrip verified'})
  (out/'retention.json').write_text(json.dumps(places,indent=2)+'\n')
  work=json.loads((out/'workload.json').read_text())['requests'][0]
  records.append({'round':block['round'],'condition':condition,**work,'retention':places})
  (r/'matrix-progress.json').write_text(json.dumps(records,indent=2)+'\n')
  print('DONE',name,'capture_ms',round(work['capture_ns']/1e6,2),'answer_ms',round(work['answer_ns']/1e6,2),'trace_GB',round(work['trace_bytes']/1e9,3),flush=True)
(r/'matrix-complete.json').write_text(json.dumps({'completed_processes':len(records),'allocation_sha256':hashlib.sha256((r/'allocation.json').read_bytes()).hexdigest()},indent=2)+'\n')
