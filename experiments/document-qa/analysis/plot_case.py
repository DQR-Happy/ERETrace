"""Three memory-layout figures from all events, with explicit visualization bins."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
r=Path(__file__).resolve().parents[1];src=r/'diagnostic-v7/extracted';out=r/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
names=['Q','K','V','Mask','Sinks','KV max','Output','Output metadata']
# Actual ggml KV allocation: context8192 * eight heads *128 half elements.
# Other pointer windows are used only as descriptive address bands, never as
# claims about accessed bytes or tensor contents.
widths=[16<<20,16<<20,16<<20,8<<20,0,0,16<<20,8<<20]
summary=json.loads((src/'summary.json').read_text());manifest=[]
for entry in summary['detail']:
 token=entry['token'];npz=src/f'dmat-token{token}.npz';a=np.load(npz);d=a['dmat'];p=a['pointers'];base=a['base'];dims=a['dims']
 valid=d['clock']>0;memory=valid&(d['addr']>0);addr=d['addr'][memory];clock=d['clock'][memory].astype(float)/1000
 group=np.full(len(addr),-1,dtype=np.int8)
 for i,(ptr,width) in enumerate(zip(p,widths)):
  if ptr and width:group[(addr>=ptr)&(addr<ptr+width)]=i
 assert not np.any(group<0),f'unclassified addresses: {np.unique(addr[group<0])[:10]}'
 pages,first_index,rank=np.unique(addr>>16,return_index=True,return_inverse=True)
 tmax=float(clock.max());edges=np.linspace(0,tmax,321)
 hist,_,_=np.histogram2d(rank,clock,bins=[np.arange(len(pages)+1)-.5,edges]);assert int(hist.sum())==len(addr)
 groups=[];page_groups=group[first_index].tolist()
 assert np.all(group==np.array(page_groups)[rank])
 for j,i in enumerate(page_groups):
  if j==0 or i!=page_groups[j-1]:groups.append((j,i))
 stage='Final prefill chunk' if token==0 else ('First decode step' if token==1 else 'Decode step 63')
 fig=plt.figure(figsize=(9.4,6.2),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[3,1.15])
 display_rows=np.arange(len(pages));spacers=12
 for j,i in groups[1:]:display_rows[j:]+=spacers
 display_hist=np.zeros((int(display_rows[-1])+1,hist.shape[1]));display_hist[display_rows]=hist
 ax=fig.add_subplot(gs[0,:]);h=ax.pcolormesh(edges,np.arange(len(display_hist)+1)-.5,np.ma.masked_equal(display_hist,0),cmap='viridis',norm=LogNorm(vmin=1,vmax=10000),rasterized=True)
 ticks=[];labels=[]
 for j,i in groups:
  end=next((jj for jj,ii in groups if jj>j),len(pages));ticks.append((display_rows[j]+display_rows[end-1])/2);labels.append(f'{names[i]} ({end-j} '+('page)' if end-j==1 else 'pages)'))
 ax.set_yticks(ticks,labels,fontsize=9);ax.set_ylabel('Observed 64 KiB virtual-address pages\n(gaps elided; regions separated for readability)');ax.set_xlabel('Time after each thread entry (instrumented kilocycles)')
 ax.set_title(f'{stage}: {int(np.prod(dims[:3]))} CTAs, {len(addr):,} memory events',loc='left',fontweight='bold',pad=9)
 cb=fig.colorbar(h,ax=ax,pad=.02);cb.set_label('Events per page / time bin\n(log scale)')
 lower=fig.add_subplot(gs[1,0]);nwarp=(int(np.prod(dims[3:6]))+31)//32;duration=base['elapsed'].reshape(-1,nwarp).max(axis=1)/1000
 xx=np.sort(duration);yy=np.arange(1,len(xx)+1)/len(xx);lower.step(xx,yy,where='post',color='#0072B2',lw=1.8);lower.set_xlabel('CTA duration (instrumented kilocycles)');lower.set_ylabel('CDF');lower.set_ylim(0,1.02);lower.grid(alpha=.15)
 q50,q95=np.quantile(duration,[.5,.95]);lower.text(.03,.94,f'p95 / median = {q95/q50:.2f}',transform=lower.transAxes,va='top',fontsize=9)
 bars=fig.add_subplot(gs[1,1]);counts=[int((group==i).sum())for i in range(len(names))];present=[i for i,c in enumerate(counts)if c];bars.bar([names[i].replace('Output metadata','Metadata')for i in present],[counts[i]/len(addr)*100 for i in present],color='#009E73',width=.7);bars.set_ylabel('Event share (%)',fontsize=9);bars.tick_params(axis='x',labelsize=8,rotation=20);bars.set_ylim(0,100);bars.grid(axis='y',alpha=.15)
 fig.suptitle('Full Qwen inference • supplemental attention diagnosis',x=.5,fontsize=12)
 basename=f'memory-layout-{token}';fig.savefig(out/f'{basename}.png');fig.savefig(out/f'{basename}.pdf');plt.close(fig)
 np.savez_compressed(out/f'{basename}-bins.npz',histogram=hist,time_edges=edges,address_pages=pages,display_rows=display_rows,page_groups=np.array(page_groups),cta_duration_kcycles=duration)
 record={'token':token,'title':stage,'source':str(npz),'source_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'memory_events':len(addr),'branch_events':int((valid&(d['addr']==0)).sum()),'event_region_counts':dict(zip(names,counts)),'observed_pages':len(pages),'time_bins':320,'color_limits':[1,10000],'time_max_kcycles':tmax,'cta_median_kcycles':q50,'cta_p95_kcycles':q95,'cta_p95_over_median':q95/q50,'pointers':dict(zip(names,[hex(int(x))for x in p])),'semantics':'Every selected-call global and cp.async event contributes to display bins. All original slots remain in raw/NPZ. Clock is thread-local elapsed, not a cross-SM global time; cp.async timestamps indicate issue, not completion. Visualization bins are not a capture summary backend.'}
 manifest.append(record)
(out/'figure-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
