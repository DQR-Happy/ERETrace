from pathlib import Path
import json,sys,hashlib,platform
r=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
s=json.loads((r/'statistics.json').read_text());means=s['means']
names={'serial_neutrino':'原始串行 raw','async_raw':'工程化异步 raw','nvcomp_cascaded':'nvCOMP Cascaded','ispa_ere':'ISPA ERE'}
rank=sorted(means,key=lambda x:means[x]['total_ms'])
text=['# 完整推理＋采集＋分析：排除额外计算校验的实测','',f"本配置下总时间实测均值最快的是 **{names[rank[0]]}**。四个独立配对区组；同一真实工程文档问答任务、同一完整捕获范围。所有总时间直接实测，未从旧 profile 扣减校验时间。",'', '| 方法 | 纯推理基准 (s) | 捕获新增耗时 (s) | 实际分析 (s) | 完整总时间 (s) | 完整Trace (GB) |','|---|---:|---:|---:|---:|---:|']
for m in names:
 x=means[m];text.append(f"| {names[m]} | {x['inference_ms']/1000:.4f} | {x['capture_increment_ms']/1000:.4f} | {x['analysis_ms']/1000:.4f} | {x['total_ms']/1000:.4f} | {x['trace_bytes']/1e9:.4f} |")
text+=['','推理和采集在现有系统中重叠。上表“捕获新增耗时”＝实际推理至Trace关闭的耗时−同区组前后Native推理均值，因此三列可加和，不对重叠时间重复计数。原始真实边界的 instrumented_execution_ms、capture_tail_ms、capture_ms 均保存在 measurements.csv；不能将新增耗时解释为单独D2H或编码时间。','', '分析包含从真实完整Trace读取、精确解码（仅编码方法需要）、每次launch的CTA时长中位数/P95/max、活跃SM和prefill/decode/operator归属统计，并写出完整分析JSON。全部方法移除了额外全数据SHA、SM/sentinel全量检查和实验identity断言；必要格式/长度/codec状态检查和受检raw回退保留。','', '## 对照差距与重复','']
for m,e in s['ere_vs_baselines'].items():
 lo,hi=e['descriptive_bootstrap_ci95_ms'];diff=e['baseline_minus_ere_mean_ms'];text.append(f"- 相对{names[m]}，ERE完整总时间均值差（基线−ERE）{diff/1000:.4f} s，均值比{e['mean_speedup']:.4f}×；ERE在{e['ere_faster_blocks']}/{e['blocks']}区组更快；配对bootstrap描述性95%差值区间[{lo/1000:.4f}, {hi/1000:.4f}] s。")
text+=['','仅4区组用于当前配置的倾向与差距判断；区间为描述性，不宣称跨任务显著性。逐区组有效结果全部保留。采集最快、分析最快和完整总时间最快是三个不同问题，本报告以最后一个排名。','', '## 完整性和性能计时','', '第一配对区组的四个正式完整Trace，在整个性能计时和模型销毁后，分别由原带校验分析器独立处理。全部实际分析JSON逐项相同；完整SHA和全数组检查通过。每个正式Trace另外在性能计时外压缩、SHA及解压roundtrip验证，完整数据按retention-index.json路径留存，不复制进本文。', '', '独立正确性不是性能分析的一部分：原 case 的生产者同Map无损检查和精确C++解码校验，见 ../neutrino-inference-case-20261013/run-20261013/correctness-audit.json 和 fast-decoder-validation.json。本次source/编码器/解码器binary SHA见manifest.json，捕获语义和编码机制未变。','', '## 实验入口与证据','', '- 预先固定协议：protocol.md；顺序：allocation.json。','- 时间/空间数据：measurements.csv、statistics.json；区组bootstrap索引：bootstrap-indices.npy。','- 3090-new：`/root/ispa-e2e-noverify-20261013/`；命令：`/root/miniconda3/bin/python -u code/execute.py`；实际每次argv/env在capture/*/command.json。','- 原始日志、侧车、完整分析及资源样本：remote-metadata.tar.zst（本地压缩快照）及同路径远端capture/。','- 完整Trace留存与原始/存档SHA：retention-index.json（host=3090-new）；计时内文件大小不采用zstd研究存档后的大小。','- 本次未更改论文正文或被冻结的主实验材料。','', '## 软件与来源','', f"后处理Python {platform.python_version()}；实验Python/NumPy/GPU/CPU quota等实际版本见manifest.json。遵循experimental-design和statistical-analysis，原GPU实现沿用optimize-for-gpu指导。技能方法引用：Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://doi.org/10.48550/arXiv.2609.00065 。",'']
(r/'report.md').write_text('\n'.join(text))
print('\n'.join(text[:11]))
