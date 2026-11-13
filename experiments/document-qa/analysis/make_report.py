"""Publish only completed real measurements; all data stays path-referenced."""
from pathlib import Path
import json,statistics
r=Path(__file__).resolve().parents[1];parent=r.parent
s=json.loads((r/'statistics.json').read_text());assert s['complete'] and s['n_blocks']==7
m=s['mean_results'];labels={'serial_neutrino':'串行 Neutrino 同 probe','async_raw':'Async raw','nvcomp_cascaded':'nvCOMP Cascaded','ispa_ere':'ISPA ERE'}
def cmp(base,metric):return next(x for x in s['comparisons']if x['baseline']==base and x['metric']==metric)
def effect(base,metric):
 x=cmp(base,metric);l,h=x['ci95_percent_reduction'];return f"{x['paired_mean_percent_reduction']:.2f}%（配对 95% CI {l:.2f}%–{h:.2f}%）"
table='| 采集后端 | 完整捕获 ms | 恢复和分析 ms | 捕获加分析 ms | 方法输出 trace GB | trace 加元数据 GB |\n|---|---:|---:|---:|---:|---:|\n'
for c in labels:
 x=m[c];table+=f"| {labels[c]} | {x['capture_ms']:.2f} | {x['analysis_ms']:.2f} | {x['answer_ms']:.2f} | {x['trace_bytes']/1e9:.3f} | {x['trace_plus_metadata_bytes']/1e9:.3f} |\n"
resource='| 后端 | 整卡 GPU 占用峰值 MiB | 进程 RSS 峰值 MiB | 采集 host 池 MiB | 采集 device 池 MiB | 额外编码 device 工作区 MiB |\n|---|---:|---:|---:|---:|---:|\n'
measurements=list(__import__('csv').DictReader((r/'measurements.csv').open()))
for c in labels:
 rows=[x for x in measurements if x['condition']==c];rss=m[c]['max_process_rss_bytes']/2**20;host=max(float(x['collector_host_capacity_bytes'])for x in rows)/2**20;device=max(float(x['collector_device_capacity_bytes'])for x in rows)/2**20;workspace=max(float(x['encoder_device_workspace_bytes'])for x in rows)/2**20
 resource+=f"| {labels[c]} | {m[c]['max_gpu_mib']:.0f} | {rss:.1f} | {host:.1f} | {device:.1f} | {workspace:.3f} |\n"
profile=json.loads((r/'consumer-profile/profile.json').read_text());native=json.loads((r/'native-context.json').read_text());correct=json.loads((r/'batch-correctness.json').read_text());fig=json.loads((r/'figures/figure-manifest.json').read_text());audit=json.loads((r/'correctness-audit.json').read_text());assert audit['complete']
ereprofile=profile['replays'][0];rawprofile=profile['replays'][1]
dev={b:json.loads((r/f'development-batch-{b}/workload.json').read_text())['requests']for b in [256,512,1024]}
devmeans={b:statistics.mean(x['capture_ns']/1e6 for x in rows)for b,rows in dev.items()}
text=f'''# 完整推理中的 FlashAttention 访存诊断与无损捕获收益

已在 3090-new 完成真实项目文档问答、28 个正式完整推理捕获请求、两种编码的独立正确性验证、三次自然 attention 调用的细访存采集，以及同一完整 Map 的消费阶段回放。主比较只更换捕获后端，使用同一推理二进制、同一完整 warp 记录、同一输入和同一分析器。

ISPA ERE 相对 nvCOMP 的捕获加分析时间减少 {effect('nvcomp_cascaded','answer_ms')}，trace 减少 {effect('nvcomp_cascaded','trace_bytes')}；相对串行采集，捕获加分析时间减少 {effect('serial_neutrino','answer_ms')}。相对 Async raw 的捕获加分析时间差为 {effect('async_raw','answer_ms')}，区间跨零，未证实稳定的完整流程时间改善；其 trace 减少 {effect('async_raw','trace_bytes')}。本例的收益是降低完整观测的获取和消费成本，原生模型调优候选未通过正确性，未计入有效推理加速。

## 现实任务与测量边界

现有Qwen3-8B Q4_K_M GGUF（内置模型名为Qwen3 8B Awq Compatible Instruct，精确版本以文件SHA标识），真实 Neutrino 项目 README 文本片段和工程问题，B1、context8192、batch/ubatch512、8 CPU 线程、全 GPU、FlashAttention 开启、Graph off、固定生成64 token。4 个开发问题用于调优，3 个独立留出问题按轮次轮换；留出输入长度2663–2665 token。无重复填充、随机输入或单 kernel 替代。

未开启捕获的三个留出完整请求分别1091.480、1092.090、1092.916ms，作为原生计算成本的描述性参考，不混入后端配对统计。

7 个配对进程轮次，轮内条件顺序由 seed20261013 冻结。每进程2次预热、1次正式完整请求；GPU 工作串行。原84请求计划因 pilot 的6.53GB完整输出和存储限制，在正式采集前修订为28请求，修订见[执行修订]({r/'execution-amendment.md'})。没有按正式结果删样本或增加轮次。统计单位为进程轮次，10000次配对 bootstrap、seed20261014；warp 和 CTA 不充当统计样本。

T_capture 从 prefill 开始，到64 token完整推理、应用同步和请求 trace 文件关闭。T_answer 再包括整个文件的精确恢复、SHA核对、全部launch的CTA持续时间统计、阶段/算子关联及分析JSON写完；归档和SSH传输在终点外。模型加载、预热和JIT不进入正式请求时间。消费阶段 nvCOMP GPU 工作暂停采集，避免递归捕获分析器；模型捕获范围没有缩小。

正式合同是每 warp 的全部16B block_sched记录：start u64、elapsed u32、SM id u32，以及帧中完整维度、顺序、launch/phase归属。三个DMAT调用属于独立的补充诊断阶段，**不在四后端主时间表中**。原始DMAT包括全部global/cp.async地址事件、branch事件和分配槽；这里没有声称ERE已适配可变长DMAT。细探针完整请求的实测捕获为46.794秒，含该阶段冷JIT；所有图的原始地址与记录均保留。

## 完整捕获与分析的性能

{table}
表中均为7进程均值。GB采用10⁹B，trace列含各格式文件头、引用表和raw回退。总文件列另加launch侧车、分析输出、统计和资源记录；侧车含统一保留的两次预热launch，主trace列仅正式请求；不含模型、编译缓存及归档副本。zstd归档只用于保存原始证据，未计入方法压缩收益。Async raw与串行raw的头部结构不同，故文件字节略有差别。

ERE 相对 nvCOMP 的完整捕获时间减少 {effect('nvcomp_cascaded','capture_ms')}，相对 Async raw 减少 {effect('async_raw','capture_ms')}。全部效应量、逐轮原始值和区间见[统计]({r/'statistics.json'})与[测量CSV]({r/'measurements.csv'})。它不能被改写为“模型原生推理提速同样百分比”。

{resource}
资源列为各独立进程运行窗口内的实测峰值均值，含模型加载、预热、捕获与消费；GPU为nvidia-smi整卡memory.used，以0.2秒采样，非进程归因值；运行时没有其他GPU任务，瞬时峰值可能漏过。采集池容量和编码工作区由runtime直接记录，不是由trace体积推算。Serial列0表示没有arena池，仍有逐launch的普通分配；所有方法还有共同模型/KV/计算缓冲及metadata。nvCOMP额外host工作区单列在测量CSV中；不要把所有host工作区都当作pinned内存。

## 从完整请求到三张访存图

独立原生CUPTI记录49,624次launch，无dropped/overflow；GPU kernel时间合计1026.436ms。量化matmul/matvec占77.68%，attention及辅助kernel占8.92%。Prefill的matmul/matvec为423.672ms，decode为373.628ms；对应attention及辅助分别43.210ms和48.351ms。因此attention局部图不能代表完整请求的主瓶颈。

全流程粗probe保留约4.08亿warp记录；其中decode matmul/matvec为279,820,800条。attention本体只有933,696条，约占0.23%。完整上下文同时揭示了计算时间的主要去向和trace的数据量去向，两者并不按同一比例增长。

### 最后一个 prefill 块

![最后一个prefill块的真实内存布局]({r/'figures/memory-layout-0.png'})

104个真实剩余输入token的第一层attention，164 CTA，7,071,232条内存事件。K、V各有360,448个不同指令地址，每个地址恰好被访问7次，三次调用的K/V不同地址集合完全一致。104个query按该kernel的ncols16分组，ceil(104/16)=7，与观测重复次数一致；这是query分组导致的K/V重复发起读取。每个K/V区域各覆盖88个64KiB页面；Q/输出各27页面，反映prefill处理较多query。

原生资源查询为35,072B dynamic shared、223寄存器/线程、最多2 CTA/SM。3090的80 SM最多同时容纳160个这种CTA，164 CTA存在末波只有4个CTA的配置条件；这是由配置推出的容量约束，不能单凭图将它说成原生FIFO调度或缓存miss的证明。

### 第一次 decode

![第一次decode的真实内存布局]({r/'figures/memory-layout-1.png'})

80 CTA，786,560条内存事件。K/V各360,448条，地址没有重复，K/V事件合计占91.65%；Q和输出各只占1页面。原生decode kernel使用66,080B dynamic shared、196寄存器/线程，资源查询只允许1 CTA/SM。两个CTA的shared需求132,160B超过sm86每SM100KiB上限，已构成明确的occupancy限制；80 CTA本身可以铺满80 SM。资源上限参见[NVIDIA Ampere指南](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)。

### 第63步 decode

![第63步decode的真实内存布局]({r/'figures/memory-layout-63.png'})

内存事件数、K/V地址集合和页面数与第1步一致，虽然逻辑KV随生成增长。这表明本窗口内实际tile访问粒度没有跨入新的地址/工作量台阶；不能据此声称更长上下文永远没有增长。带细probe的CTA p95/median分别为1.122、1.102、1.167，没有给出原文同量级FIFO归因所需的证据。这些是插桩持续时间，不替代原生kernel latency。

三张图使用所有观测事件形成64KiB页面×320时间箱，统一对数色阶1–10000；纵向去掉未访问地址间隙并插入区域间留白，仅为了阅读。原始逐线程全部槽位仍在bin/NPZ及完整请求归档中。横轴是每线程entry后的clock差，不能把不同SM的clock拼成统一全局时轴；cp.async后取时标记发起进度，不是内存传输完成或cache miss。语义参见[NVIDIA PTX cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。图数据和精确地址计数见[图清单]({r/'figures/figure-manifest.json'})、[地址诊断]({r/'address-diagnosis.json'})。

## 收益来自哪里 又被什么抵消

四条件保持相同完整记录和计算任务。串行条件逐launch分配、同步、回传，延迟大；arena把完整记录批量异步回传。ERE进一步降低实际传输和文件字节，从执行参照精确恢复字段，超范围帧回退raw；未消除warp/事件、未替换为查询摘要。nvCOMP使用同一arena、SOA Cascaded、64KiB chunk和复用解压workspace，避免用逐chunk反复分配的弱实现制造优势。

消费端是新的主要成本。同一份ERE完整Map恢复成raw后，两种输入生成的整个分析答案SHA完全一致。cProfile回放：ERE消费{ereprofile['profiled_wall_ns']/1e9:.3f}秒，raw消费{rawprofile['profiled_wall_ns']/1e9:.3f}秒；ERE解码函数累计约3.11秒，SHA核对各约4.05秒，NumPy规约约2.8–3.1秒。这解释了为何ERE捕获明显更快、空间明显更小，但T_answer相对raw的优势被CPU恢复和完整扫描抵消。profile有测量开销，只用于机制解释，不替代正式时间表。见[消费阶段证据]({r/'consumer-profile/profile.json'})。

## 已尝试的模型调优及判定

真实完整推理开发集的batch256/512/1024均值分别{devmeans[256]:.3f}/{devmeans[512]:.3f}/{devmeans[1024]:.3f}ms。1024相对512约快{100*(devmeans[512]-devmeans[1024])/devmeans[512]:.2f}%，但自由生成token出现差异。固定同一512生成前缀再比较4×64×151936 logits，最大绝对误差{correct['max_absolute']:.6f}，RMSE{correct['rmse']:.6f}，{correct['exceed_tolerance']:,}/{correct['total']:,}个值超出事前abs1e−3+rel1e−3门槛，无非有限值。故正式捕获固定512，没有把这个候选记成有效修复或放宽门槛。

本case支持更低成本的完整诊断，支持用完整上下文约束优化优先级，并发现/修复了影响可信观测的真实探针缺陷；它不支持“已实现FlashAttention修复后的完整推理加速”。下一步应优先研究消费端全记录解码与分析融合，或针对本case实际主导的量化matmul/matvec作正确性受约束的优化。任何后续方案仍须逐条精确恢复原始记录。

## 正确性与复现入口

28个正式进程全部输入和64生成token与对应原生留出请求一致，捕获errors/excluded/original/pruned均为零；各配对轮次的launch名称、维度、phase、记录数和顺序digest一致。两种编码的独立完整请求验证分别核对18,597个ERE待编码帧、49,624个nvCOMP帧的同次GPU Map。ERE C++与独立NumPy decoder另对全部18,067个实际编码帧、401,573,568条记录逐字节一致，其余帧原样保留。所有正式完整trace和细probe完整请求均做归档恢复SHA检查。详见[正式正确性审计]({r/'correctness-audit.json'})、[解码器审计]({r/'fast-decoder-validation.json'})、[细probe留存]({r/'diagnostic-v7/retention.json'})。

修复都放在隔离目录：非阻塞stream计数回放增加正确stream同步；在内存指令前保存地址并在其后读时间；DSL不发射literal赋值，分支零标记通过显式算术生成；opcode匹配支持L2::修饰符，从而捕获K/V cp.async。v3–v5无效地址与v6缺少cp.async覆盖的数据不用于最终图或结论；v7才是最终诊断。失败日志与生成PTX保留，没有用删除错误地址来美化图。见[实验问题与修复]({r/'experimental-issues.md'})。

本地实现与证据根目录：`{r}`。远端：`3090-new:/root/autodl-tmp/neutrino-inference-case-20261013/`，部分完整归档位于`/root/neutrino-case-archives-20261013/`；每个正式条件的retention.json给出实际保存主机、绝对路径与SHA。源码、二进制和模型版本采用SHA清单，无Git元数据时不虚构commit；见[最终版本清单]({r/'final-provenance.json'})。

运行入口：先用`code/run_one.py <condition> <round>`执行一次完整任务；`code/execute_matrix.py`按allocation.json控制GPU串行与精确归档；`code/analyze_matrix.py`核对并统计；`code/plot_case.py`从diagnostic-v7的原始数组生成三图。完整实际argv和环境开关在各条件command.json，细探针与构建命令在[执行协议]({parent/'protocol-executed.md'})。原始代码、日志及完整数据不粘贴到本文。

NeutriNo原文依据为[本地OSDI2025论文](/Users/dengqirui/Documents/BITS/neutrino/papers/osdi25-huang-songlin.pdf)§7：原文以DMAT、block_sched和branch进度分析attention，再用硬件profiler核查。本case沿用从粗到细的证据链，加入完整请求上下文、输出正确性和捕获消费成本，不搬用其A100的FIFO/cache归因。官方页面：[NeutriNo OSDI2025](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)。冻结论文证据目录和论文正文未修改。
'''
(parent/'case-report.md').write_text(text)
print(parent/'case-report.md')
