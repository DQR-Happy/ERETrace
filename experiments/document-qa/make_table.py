"""Exact additive capture-only decomposition from existing raw timestamps.
No GPU experiments or method-specific analysis task. All valid 4 blocks retained.
"""
from pathlib import Path
import json,csv,hashlib,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R=Path(__file__).resolve().parent
PARENT=R.parent/'ispa-e2e-noverify-20261013'
METHODS=['native','serial_neutrino','async_raw','nvcomp_cascaded','ispa_ere']
NAMES={'native':'Native','serial_neutrino':'Serial raw','async_raw':'Async raw (arena)','nvcomp_cascaded':'nvCOMP Cascaded','ispa_ere':'ISPA ERE'}
CN={'native':'无捕获 Native','serial_neutrino':'原始串行 raw','async_raw':'异步 raw（同池）','nvcomp_cascaded':'nvCOMP Cascaded','ispa_ere':'ISPA ERE'}
KEYS=['prefill_ms','model_steps_ms','token_control_ms','tail_ms','capture_ms']
rows=[];sources=[]
for method in METHODS:
 for block in range(1,5):
  tags=['native_before','native_after'] if method=='native' else [method]
  requests=[]
  for tag in tags:
   p=R/'inputs/capture'/f'r{block}-{tag}'/'workload.json'
   payload=json.loads(p.read_text());x=payload['requests'][0];sources.append(p)
   assert x['output_tokens']==64 and len(x['decode_steps_ns'])==63
   assert x['prefill_ns']+x['decode_ns']==x['inference_ns']
   assert x['inference_ns']+x['drain_ns']==x['capture_ns']
   steps=sum(x['decode_steps_ns']);control=x['decode_ns']-steps
   assert steps>=0 and control>=0
   requests.append({'prefill_ms':x['prefill_ns']/1e6,'model_steps_ms':steps/1e6,
     'token_control_ms':control/1e6,'tail_ms':0.0 if method=='native' else x['drain_ns']/1e6,
     'capture_ms':(x['inference_ns'] if method=='native' else x['capture_ns'])/1e6,
     'trace_bytes':x['trace_bytes'],'frames':None if method=='native' else 49624,
     'prompt_id':x['prompt_id'],'input_tokens':x['input_tokens'],'input_ids':x['input_ids'],'output_ids':x['output_ids']})
  assert all(x['input_ids']==requests[0]['input_ids'] and x['output_ids']==requests[0]['output_ids'] for x in requests)
  row={'block':block,'method':method,'prompt_id':requests[0]['prompt_id'],
       **{k:float(np.mean([x[k]for x in requests]))for k in KEYS+['trace_bytes']}}
  assert abs(sum(row[k]for k in KEYS[:4])-row['capture_ms'])<1e-8
  rows.append(row)

parent=json.loads((PARENT/'statistics.json').read_text())
means={};sd={}
for method in METHODS:
 selected=[x for x in rows if x['method']==method]
 means[method]={k:float(np.mean([x[k]for x in selected]))for k in KEYS+['trace_bytes']}
 sd[method]={k:float(np.std([x[k]for x in selected],ddof=1))for k in KEYS}
 if method!='native':
  assert abs(means[method]['capture_ms']-parent['means'][method]['capture_ms'])<1e-8
  assert means[method]['trace_bytes']==parent['means'][method]['trace_bytes']
 assert abs(sum(means[method][k]for k in KEYS[:4])-means[method]['capture_ms'])<1e-8

# Descriptive paired bootstrap, common block index for all stages.
seed=2026101319;idx=np.random.default_rng(seed).integers(0,4,size=(10000,4))
np.save(R/'bootstrap-indices.npy',idx.astype('uint8'))
effects=[]
for baseline in METHODS[1:-1]:
 total_saving=means[baseline]['capture_ms']-means['ispa_ere']['capture_ms']
 for key in KEYS:
  v=np.array([x[key]for x in rows if x['method']==baseline])-np.array([x[key]for x in rows if x['method']=='ispa_ere'])
  effects.append({'baseline':baseline,'stage':key,'mean_saved_ms':float(v.mean()),
    'descriptive_paired_ci95_ms':np.quantile(v[idx].mean(axis=1),[.025,.975]).tolist(),
    'ere_faster_blocks':int((v>0).sum()),'paired_saved_ms':v.tolist(),
    'share_of_total_saving_pct':float(v.mean()/total_saving*100)})

with (R/'phase-measurements.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (R/'phase-means.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['method']+KEYS+['capture_sd_ms','trace_GB','ere_speedup_vs_method'])
 w.writeheader()
 for m in METHODS:
  w.writerow({'method':m,**{k:means[m][k]for k in KEYS},'capture_sd_ms':sd[m]['capture_ms'],
    'trace_GB':means[m]['trace_bytes']/1e9,
    'ere_speedup_vs_method':None if m=='native' else means[m]['capture_ms']/means['ispa_ere']['capture_ms']})

result={'source_study':'3090-new document QA, existing 4 complete paired blocks; 16 captured requests plus 8 native controls',
 'new_experiments_run':False,'primary_end':'request inference starts to full trace drain/file close, no offline analysis/extra verification/archive',
 'means_ms':means,'sample_sd_ms':sd,'paired_ere_savings':effects,
 'phase_definitions':{'prefill_ms':'begin -> qwen_prefill return, including request mark/probes and interleaved collection',
 'model_steps_ms':'sum of 63 independently recorded qwen_step wall intervals; model generation and overlapping instrumentation/collection, not pure GPU compute',
 'token_control_ms':'decode_ns - sum(decode_steps_ns), including 64 logits retrievals/greedy selections, token/request marks and any waits, loop and timestamp overhead; derived aggregate, not isolated CPU/transfer time',
 'tail_ms':'model generation complete -> capture boundary return; remaining synchronization, drain and file close; for Native no capture stage',
 'capture_ms':'sum of above 4 windows, original capture endpoint; native uses inference end'},
 'no_component_times_inferred':['encoding','D2H','writer work (these overlap and need dedicated diagnostics)'],
 'native_aggregation':'mean of before/after controls within each block, then mean/SD across 4 blocks',
 'baseline_note':'This case async raw uses the same arena infrastructure as ERE. Main paper +Async uses 32 reusable slots. Do not identify them as identical implementations or mix 256-token main results with 2663-2665-token case.',
 'bootstrap':{'draws':10000,'seed':seed,'unit':'paired process block','interval':'exploratory percentile 95%, n=4, not general significance'},
 'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
 'sources':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources+[PARENT/'statistics.json',R/'make_table.py']}}
(R/'phase-breakdown.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')

md=['# 完整推理＋采集的阶段分解','',
 '同一真实文档问答：RTX3090、Qwen3-8B Q4、2663–2665输入/64输出token、B1、context8192；使用刚完成的4个配对区组，不新增GPU实验。每个捕获请求49,624个launch、约4.08亿条完整原始记录；输出token与捕获scope已在来源实验逐区组核对一致。计时结束于Trace关闭，不含离线分析、额外校验或实验存档。', '',
 '| 方法 | Prefill ms | 63次模型step ms | 生成控制/边界 ms | 末尾排空/关闭 ms | 完整推理＋采集 ms | Trace GB | ERE相对此行加速 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for m in METHODS:
 x=means[m]; speed='—' if m=='native' else f"{x['capture_ms']/means['ispa_ere']['capture_ms']:.2f}×"
 vals=[f'{x[k]:.2f}'for k in KEYS[:3]]+[('—' if m=='native' else f"{x['tail_ms']:.2f}"),f"{x['capture_ms']:.2f}",('—'if m=='native' else f"{x['trace_bytes']/1e9:.3f}"),speed]
 md.append('| '+CN[m]+' | '+' | '.join(vals)+' |')
md+=['', '**阶段解释：**前两阶段包含模型执行期间重叠的探针、编码/回退、回传、写出及等待。生成控制/边界由已有decode总时长减去63次step时长得到，包括logits回传、greedy选词、mark、循环及相关等待；不是独立的纯CPU或D2H时长。四个阶段在每个请求上严格相加等于完整捕获时间。Native无捕获尾部，使用其实际推理完成时间。', '',
 '总时间样本SD（ms，4个区组；Native每区组前后均值）：'+ '；'.join(f"{CN[m]} {sd[m]['capture_ms']:.2f}"for m in METHODS)+'。', '',
 '## ERE的时间节省分布', '',
 '| 相对基线 | Prefill节省 ms | 模型step节省 ms | 控制/边界节省 ms | 末尾节省 ms | 总节省 ms | Trace减少 |',
 '|---|---:|---:|---:|---:|---:|---:|']
for m in METHODS[1:-1]:
 vals=[f"{means[m][k]-means['ispa_ere'][k]:.2f}"for k in KEYS]
 vals+=[f"{100*(1-means['ispa_ere']['trace_bytes']/means[m]['trace_bytes']):.2f}%"]
 md.append('| '+CN[m]+' | '+' | '.join(vals)+' |')
md+=['', '阶段节省是应用墙钟预算的分解，不能解释为独立编码、D2H或写线程成本，不能称为原生模型计算加速。逐区组数据、配对描述性区间、来源SHA及精确定义见phase-breakdown.json / phase-measurements.csv。', '',
 '当前长文档case的异步raw采用与ERE同一arena；原论文256-token主实验的+Async采用32个可复用slots，二者不要混称同一具体实现。本文只汇总本次4区组数据；未替换论文中先前7区组的case表。', '',
 '## 可复用文件', '', '- capture-breakdown.pdf / .png：排版表。','- capture-table.tex：可直接放入table*的booktabs表，英文标签对应本文定义。','- phase-means.csv / phase-measurements.csv / phase-breakdown.json：完整数据与来源。','- make_table.py：生成入口，命令 `.venv-aiperf/bin/python reports/ispa-capture-breakdown-20261013/make_table.py`。', '',
 '方法与软件：statistical-analysis用于区组均值、配对效应与小样本描述性区间；scientific-visualization用于不删样本、统一量纲和表格排版。技能引用沿用本次来源protocol.md的Kassis et al. (2026), https://doi.org/10.48550/arXiv.2609.00065 。','']
(R/'report.md').write_text('\n'.join(md))

# Publication-ready vector table; keep measured values unchanged.
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(11.8,4.0),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.035,.93,'Complete inference + capture',fontsize=17,fontweight='bold',color='#122539',va='center')
ax.text(.035,.86,'RTX 3090 · Qwen3-8B Q4 · document QA · 2663–2665 input / 64 output tokens',fontsize=9,color='#46596a')
widths=np.array([.205,.098,.115,.112,.107,.138,.095,.13]);widths=widths/widths.sum()*.93
edges=np.r_[.035,.035+np.cumsum(widths)]
headers=['Collector','Prefill\n(ms)','63 decode steps\n(ms)','Token control\n(ms)','Drain / close\n(ms)','Total ± SD\n(ms)','Trace\n(GB)','ERE speedup\nvs. row']
ax.add_patch(Rectangle((.035,.705),.93,.095,facecolor='#122539',edgecolor='none'))
for j,h in enumerate(headers):
 ax.text(edges[j]+(.012 if j==0 else widths[j]/2),.752,h,ha='left'if j==0 else 'center',va='center',fontsize=9,color='white',fontweight='bold')
row_h=.093
for i,m in enumerate(METHODS):
 y=.705-(i+1)*row_h
 ax.add_patch(Rectangle((.035,y),.93,row_h,facecolor='#dcecf8'if m=='ispa_ere' else ('#f2f5f8'if i%2==0 else 'white'),edgecolor='none'))
 x=means[m];vals=[NAMES[m]]+[f'{x[k]:,.2f}'for k in KEYS[:3]]+['—'if m=='native' else f"{x['tail_ms']:.2f}",f"{x['capture_ms']:,.2f} ± {sd[m]['capture_ms']:.2f}",'—'if m=='native' else f"{x['trace_bytes']/1e9:.3f}",'—'if m=='native' else f"{x['capture_ms']/means['ispa_ere']['capture_ms']:.2f}×"]
 for j,v in enumerate(vals):
  ax.text(edges[j]+.012 if j==0 else edges[j+1]-.012,y+row_h/2,v,ha='left'if j==0 else 'right',va='center',fontsize=9.2 if j!=5 else 8.6,color='#122539',fontweight='bold'if m=='ispa_ere' else 'normal')
 ax.plot([.035,.965],[y,y],color='#c8d4de',linewidth=.55)
ax.text(.035,.19,'Means over 4 paired process blocks; SD across block values. Native: before/after controls averaged within each block.',fontsize=8.0,color='#46596a')
ax.text(.035,.135,'Prefill / decode steps include overlapping collection. Token control includes logits, greedy selection, marks and waits.',fontsize=8.0,color='#46596a')
ax.text(.035,.08,'End: full trace closed. Offline analysis, additional verification and experiment archives excluded. All full records retained.',fontsize=8.0,color='#46596a')
fig.savefig(R/'capture-breakdown.pdf',facecolor='white')
fig.savefig(R/'capture-breakdown.png',dpi=200,facecolor='white')
plt.close(fig)
tex=[r'\begin{table*}[t]',r'\centering',r'\caption{Document QA: complete inference and capture, decomposed by application timing windows. Means of four paired process blocks; all times are milliseconds. Prefill and model steps include overlapping collection. Token control includes logits retrieval, greedy selection, marks and related waits. Offline analysis and additional verification are excluded. Trace volume is decimal GB.}',r'\label{tab:case-capture-phases}',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}lrrrrrrr@{}}',r'\toprule',r'Collector & Prefill & Model steps & Token control & Drain/close & Total & Trace (GB) & ERE speedup \\',r'\midrule']
for m in METHODS:
 x=means[m];vals=[NAMES[m].replace('nvCOMP Cascaded','+nvCOMP').replace('Async raw (arena)','+Async (arena)')]+[f'{x[k]:.2f}'for k in KEYS[:3]]+['--'if m=='native' else f"{x['tail_ms']:.2f}",f"{x['capture_ms']:.2f}",'--'if m=='native' else f"{x['trace_bytes']/1e9:.3f}",'--'if m=='native' else f"{x['capture_ms']/means['ispa_ere']['capture_ms']:.2f}"+r'$\times$']
 if m=='ispa_ere':vals=[r'\textbf{'+v+'}'for v in vals]
 tex.append(' & '.join(vals)+r' \\')
tex +=[r'\bottomrule',r'\end{tabular}',r'\par\vspace{2pt}',r'\footnotesize ERE speedup is the row collector\textquotesingle s total time divided by ERE time. Model steps sum 63 recorded decode calls; token control is the remaining generation interval. Native has no trace drain.',r'\end{table*}']
(R/'capture-table.tex').write_text('\n'.join(tex)+'\n')
manifest=[]
for p in sorted(R.glob('*')):
 if p.is_file():manifest.append({'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(R/'manifest.json').write_text(json.dumps({'files':manifest,'all_24_requests_checked':True,'no_new_gpu_experiment':True},indent=2)+'\n')
print('\n'.join(md[:12]))
