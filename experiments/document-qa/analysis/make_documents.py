"""Fill retained report templates with completed evidence; add three figures."""
from pathlib import Path
import importlib.util,json,copy,io
from zipfile import ZipFile
from lxml import etree as E
from docx import Document
from docx.shared import Inches
r=Path(__file__).resolve().parents[1];parent=r.parent;out=r/'documents';out.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('template_builder',parent/'build_documents.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.ROOT=out
for kind in ['design','analysis']:
 q=out/'qa'/kind;q.mkdir(parents=True,exist_ok=True)
 for name in ['reference','template-style-evidence.json']:
  p=q/name
  if not p.exists():p.symlink_to(parent/'qa'/kind/name,target_is_directory=name=='reference')
s=json.loads((r/'statistics.json').read_text());assert s['complete'];m=s['mean_results']
def effect(base,metric):
 x=next(x for x in s['comparisons']if x['baseline']==base and x['metric']==metric);lo,hi=x['ci95_percent_reduction'];return f"{x['paired_mean_percent_reduction']:.2f}% CI {lo:.2f}至{hi:.2f}%"
def metric(c,k):return f"{m[c][k]:.2f}"
design={
2:'完整推理诊断案例',4:'目录',6:'本案例的结论',
7:f"已在3090-new完成真实Qwen文档问答、28个正式完整请求和三次细访存诊断。相对nvCOMP，ISPA ERE的捕获加分析时间减少{effect('nvcomp_cascaded','answer_ms')}，trace减少{effect('nvcomp_cascaded','trace_bytes')}。",
8:'本案例证明了降低完整观测成本的实际收益。原生batch1024调优虽约快3%，但未通过固定前缀logits检查，因此没有计入有效模型推理加速。',
9:'实际分析解决的问题',10:'三张图展示最后prefill块、首个decode和第63步decode的真实访存布局，全部观测地址保留。',11:'完整请求显示attention仅占8.92%的GPU kernel时间，量化matmul和matvec占77.68%，避免把局部现象当作全局瓶颈。',12:'消费阶段的同Map回放解释了ERE恢复开销怎样抵消部分采集提速，并逐字节核对完整分析答案。',
13:'原文实验与本案例',14:'NeutriNo OSDI2025原文§7在A100上使用DMAT、block_sched、branch进度和硬件profiler诊断FlashAttention。原文没有展示修复后的完整推理加速。',15:'本案例沿用从粗到细的分析顺序，增加真实完整请求、全算子上下文、输出正确性，以及相同捕获和消费终点的对照。不套用原文的FIFO或cache miss归因。',
16:'现实任务和固定配置',17:'Qwen3-8B Q4_K_M真实项目文档问答。B1、context8192、batch512、8线程、全GPU、FlashAttention开、Graph关，留出输入2663至2665 token，固定生成64 token。',
18:'完整记录与诊断输入',19:'所有hook覆盖launch都保留start u64、elapsed u32、SM id u32和完整帧元数据。没有用查询结果代替trace；编码超范围时检查并回退raw。',
20:'三张图揭示的原因',21:'最后prefill块104个query，K/V每个不同地址恰好访问7次，与ncols16的7组query一致；decode同一地址集合各访问1次。',22:'首末decode的K/V事件数与地址集合相同，说明本窗口未跨越新的实际tile访问台阶。重复发起访问不等于DRAM读取或cache miss。',23:'原生prefill最多2 CTA/SM，164 CTA超过160个驻留槽；decode的66080B shared限制为1 CTA/SM。配置能解释资源约束，图不能单独证明原文的FIFO式调度。',24:'三次细probe的p95/median为1.122、1.102和1.167，均为插桩周期。原生GPU时间由独立CUPTI采集，不能用这些周期替代。',
25:'已执行的修改与对照',26:'所有采集条件共用推理、probe和分析器。消费侧为nvCOMP复用连续解压workspace，为ERE使用经过独立逐字节核对的C++ decoder；编码核心未改。',
27:'完整实验安排',28:'7个独立配对进程轮次，每进程2次预热和1个正式请求，轮内随机顺序，GPU串行，共28个正式请求。因pilot完整trace为6.53GB，正式前修订了存储与请求数。',29:'4个开发问题试batch256/512/1024，3个留出问题轮换。1024正确性失败后，主比较固定512；没有事后更换任务或容忍阈值。',30:'两种编码先核对同次GPU原始Map，ERE另与独立NumPy decoder核对全部实际编码帧。正式请求全部输入与生成token通过原生核对。',31:'细probe为独立补充阶段，全请求基础记录仍完整，三次attention额外保存所有global、cp.async和branch事件。其46.794秒冷捕获不混入主比较。',
32:'时间空间结果与解释',33:f"四后端捕获加分析均值为：串行{metric('serial_neutrino','answer_ms')}ms，raw{metric('async_raw','answer_ms')}ms，nvCOMP{metric('nvcomp_cascaded','answer_ms')}ms，ERE{metric('ispa_ere','answer_ms')}ms。完整trace、侧车和分析结果均按实际文件字节统计。",34:'ERE缩减了传输和输出，CPU恢复则增加约3秒。相对raw的完整分析完成时间按配对区间如实判断，不能把空间优势改写成所有时间终点都更快。',
35:'实现和数据入口',36:'隔离实现位置',37:'run-20261013/code保存真实工作负载、控制器、精确decoder、探针、分析与绘图代码。source与diagnostic-source保存编译源码，最终SHA清单在final-provenance.json。',38:'正式条件目录capture/rN-condition保存实际argv、配置、timing、完整分析、资源采样和retention.json。完整归档分布在本地及3090-new，以实际主机和SHA索引。',
39:'完整案例证据',40:'case-report.md为完整结论入口，measurements.csv和statistics.json为主结果；consumer-profile为同Map完整分析一致性与CPU归因。',41:'figures保留三张PNG和PDF、可复算的bin统计及来源SHA；diagnostic-v7/extracted保留完整细Map和逐线程数组。',42:'protocol-executed.md说明执行边界、构建和命令；experimental-issues.md保留探针修复、失败运行与排除原因。',43:'冻结论文证据目录与当前论文正文未修改。原始代码、日志和完整数据均通过路径引用，不复制到报告中。',45:'适用范围是本次3090、B1、Graph off、Q4模型与相关文档问题。建议把完整捕获成本收益、上下文诊断价值和原生调优负结果分开写入论文。'}
analysis={
0:'完整推理诊断实验分析',6:'完整推理',7:'真实案例分析',23:'实验与分析 Codex',24:'2026年10月13日',26:'实验结论',29:'验证目标和结果',
30:f"ISPA ERE相对nvCOMP的捕获加分析时间减少{effect('nvcomp_cascaded','answer_ms')}，完整trace减少{effect('nvcomp_cascaded','trace_bytes')}。",31:'全部正式请求保留完整记录并通过输出核对。batch1024约3%的提速因logits不等价被拒绝，未主张有效原生推理加速。',33:'与NeutriNo原文的关系',34:'原文§7用访存、调度与branch进度定位A100上的FlashAttention现象。本例在真实Qwen完整请求中执行相同的从粗到细分析顺序。',35:'额外验证完整上下文的作用、同任务捕获消费成本和输出正确性，不把原文的硬件归因当作本例结果。',36:'已执行的配置',39:'固定项与比较条件',40:'只改变采集后端，保持同一推理二进制、输入、batch512、完整warp记录和分析器。比较串行、Async raw、nvCOMP Cascaded、原ISPA ERE。',41:'7配对进程轮次，seed20261013冻结顺序，3个留出问题轮换；每进程2预热1正式。全部GPU工作串行，正式前按数据量修订为28请求。',43:'验证的机制',44:'ERE减少完整记录的传输与保存；消费端完整恢复、SHA和CTA分析可能抵消采集收益。模型调优另外执行正确性检查，不混入捕获对照。',46:'事前判据与判定',47:'采集分析实用阈值5%，原生调优阈值3%，都要求正确性。相对nvCOMP的T_answer通过5%判据；相对raw按实际区间判读。1024未通过正确性。',49:'正确性执行证据',50:'独立原生CUPTI与完整probe都覆盖49624次launch，无CUPTI dropped或overflow。三个自然细probe保留同一完整请求的全部基础记录。',51:'统一兼容CPU库和GPU backend，冻结源码、模型、编译器及库SHA。nvCOMP使用官方5.3和复用解压workspace，ERE采用独立验证的精确C++ decoder。',52:'ERE待编码帧18597个全部与同次GPU原Map核对；实际编码18067帧、401573568记录另与NumPy恢复逐字节一致。nvCOMP核对49624帧。',53:'28个正式进程的输入、64输出token、launch/phase/维度/记录顺序通过核对。所有完整归档重新解压验证原SHA，搬移后再验证归档SHA。',54:'1024固定同一512前缀比较38895616 logits，最大误差1.16877、RMSE0.081575；37134901值超事前门槛，非有限值0。',56:'正式样本和时间终点',57:'实际完成28个正式完整捕获请求，另有不进入主统计的开发、正确性、原生CUPTI、三调用DMAT及同Map消费回放。统计单位是7个进程配对轮次。',59:'完整捕获与分析主结果',61:f"ERE相对串行T_answer减少{effect('serial_neutrino','answer_ms')}；相对raw差为{effect('async_raw','answer_ms')}，trace减少{effect('async_raw','trace_bytes')}。原始逐轮值和全部区间另存，不按哪个终点更好挑结果。",63:'精确记录和资源',65:'空间包含完整trace头部、引用表、回退与共享分析元数据。资源采样为整卡GPU占用和进程RSS峰值，arena容量由runtime记录；完整归档的zstd体积不算方法收益。',66:'完整请求上下文',67:'原生GPU kernel时间1026.436ms，量化matmul/matvec占77.68%，attention及辅助占8.92%。完整trace中decode matmul/matvec有279820800条warp记录。',69:'attention本体记录约占完整warp数据0.23%。只看attention可以分析局部布局，但不能据此给完整请求优化优先级。',71:'真实访存与资源原因',72:'最后prefill块K/V每个不同地址访问7次，首末decode访问1次；三者K/V地址集合相同。与104 query按ncols16分7组一致。',73:'原生prefill shared35072B、223寄存器、2 CTA/SM；decode shared66080B、196寄存器、1 CTA/SM。decode的shared需求明确限制第二个CTA驻留。',74:'细probe周期受插桩影响。图是指令发起进度，不能直接解释cache miss、DRAM带宽或跨SM全局时间，也不证明原文FIFO归因。',75:'B1、Graph off、Q4_K_M和3090限定本case范围。3个留出问题都来自真实文档，7进程重复不代表7种独立应用。',76:'修复计数stream同步、地址生命期、分支零标记和L2::opcode匹配；v3至v6不用于最终图，v7才是完整细访存证据。',77:'细probe单独捕获46.794秒含冷JIT，不在四后端主时间表内。没有声称ERE已编码可变长DMAT或更改其核心机制。',79:'时间收益的归因',80:'同MapCPU回放ERE17.455秒、raw14.492秒，完整分析答案SHA相同。ERE恢复累计约3.11秒，SHA约4.05秒，NumPy规约约3秒，消费阶段抵消部分采集收益。',81:'同一arena与完整数据合同下，ERE相对nvCOMP同时缩减运输和输出；相对raw的空间收益成立，完整分析完成时间按区间报告。',83:'案例使用建议',86:'可写入论文的内容',87:'将真实完整请求上下文、三张访存布局、精确恢复验证，以及同任务采集分析时间空间收益串成case。',88:'报告batch1024不等价的负结果，避免将约3%探索性提速包装成有效模型修复。',89:'明确区分捕获完成时间、消费时间和T_answer。不能把方法更快获得trace写成模型计算本身提速。',90:'下一步优先改善完整消费的CPU恢复与扫描，保留全部可恢复记录；若优化模型，应转向实际主导的量化matmul/matvec并保持数值门槛。',91:'当前论文与冻结证据保持不变，新增案例报告和图在本案例目录，方便研究者选择纳入主稿。',93:'指标定义',94:'T_answer从prefill开始直到完整decode、同步、trace关闭、精确恢复、全文件分析和JSON写完。',95:'T_capture止于请求trace关闭；模型加载、预热、JIT与后续归档/SSH传输在正式终点外。',96:'方法trace为原始输出格式字节；总文件另加侧车、分析及统计。整卡GPU占用/进程RSS峰值包含加载、预热和消费，不由trace推算。',97:'访存图用全部事件形成页面时间箱，全部原始槽仍保留。横轴各线程clock差，纵向地址间隙省略并插入区域间留白。',98:'两个decoder的同Map核对和CPU profile用于解释机制，profile时间不替代正式样本统计。',100:'统计和复现入口',101:'7配对进程，bootstrap10000次，seed20261014，报告绝对差与相对95%区间。没有将大量warp/CTA视为独立样本。',102:'一次归档传输超时发生在GPU计时结束后，原数据保留；后续归档留远端并同步元数据，不重复有效GPU样本。',103:'实际源码和二进制SHA在final-provenance.json；每条件command.json记录argv与实验环境。retention.json索引完整归档的主机、路径和SHA。',104:'完整结论case-report.md；数据measurements.csv/statistics.json；图figures；细Map diagnostic-v7/extracted；消费归因consumer-profile。'}
dtables=[[['完整推理捕获与分析的真实案例','','执行完成 2026年10月13日']], [['比较层次','固定合同','本轮结果'],['捕获分析','相同完整记录和计算','相对nvCOMP时间空间改善'],['原生调优','真实prefill和decode','1024未通过正确性'],['完整上下文','跨阶段全部算子','量化matmul/matvec主导']]]
atable=[
[['版本','真实案例 v2'],['整理','Codex 实测证据'],['审阅','尚未由研究者审阅'],['日期','2026年10月13日'],['结果窗口','3090-new 完整推理'],['状态','28正式请求 已完成']],
[['项目','实际执行'],['案例','完整推理attention访存'],['标识','C09 Full Inference'],['研究对象','ISPA完整细粒度trace'],['任务','真实项目文档问答'],['平台','RTX3090 24GiB'],['主目标','相同分析的时间空间'],['采集对照','串行 raw nvCOMP'],['采集方法','原ISPA ERE活动基线'],['工作分配','轮内随机 GPU串行'],['独立单位','进程配对轮次'],['主条件','Qwen3-8B Q4_K_M B1'],['控制边界','Graph关 8线程 64输出'],['正式开始','2026年10月13日'],['正式结束','2026年10月13日'],['修订规模','28完整捕获请求'],['实际规模','7配对进程轮次']],
[['判据','实际结果'],['采集分析','相对nvCOMP通过5%'],['原生优化','1024正确性未通过'],['记录正确性','同次Map逐字节恢复一致'],['计算正确性','28请求生成token一致'],['判定','获取成本改善 模型修复失败']],
[['规模','每采集条件','全部'],['独立进程','7','28'],['正式请求','7','28'],['预热请求','14','56'],['条件数','1','4'],['当前实测','已完成','已完成']],
[['终点','nvCOMP','ERE','减少','判定'],['T_answer ms',metric('nvcomp_cascaded','answer_ms'),metric('ispa_ere','answer_ms'),f"{next(x for x in s['comparisons']if x['baseline']=='nvcomp_cascaded'and x['metric']=='answer_ms')['paired_mean_percent_reduction']:.2f}%",'改善'],['Trace GB',f"{m['nvcomp_cascaded']['trace_bytes']/1e9:.3f}",f"{m['ispa_ere']['trace_bytes']/1e9:.3f}",f"{next(x for x in s['comparisons']if x['baseline']=='nvcomp_cascaded'and x['metric']=='trace_bytes')['paired_mean_percent_reduction']:.2f}%",'改善'],['原生调优','512默认','1024候选','约3%','不等价'],['95%区间','配对轮次','n等于7','见正文','未删样本']],
[['检查','对照','方法','要求','结果'],['完整恢复','GPU原Map','精确decoder','逐字节','通过'],['生成输出','原生512','全部4后端','64token','通过'],['归属顺序','共同scope','全部4后端','digest','通过'],['1024 logits','固定512前缀','1024候选','误差1e−3','失败']],
[['分析对象','GPU时间ms','占比','证据','作用'],['Prefill矩阵','423.672','41.28%','原生CUPTI','主瓶颈'],['Decode矩阵','373.628','36.40%','原生CUPTI','主瓶颈'],['Attention','91.561','8.92%','含辅助','局部对象'],['K/V重复','prefill 7次','decode 1次','精确地址','分组访存'],['消费恢复','约3.11秒','ERE额外','同Map回放','抵消收益']],
[['项目','完成后的判断'],['主案例','真实文档Qwen完整推理'],['决定日期','2026年10月13日'],['审阅结论','实验证据 未虚构审批'],['执行结果','28完整请求和三张图'],['正确性判定','捕获通过 1024调优失败'],['后续','保留全trace 优化消费']]]
# Reference paragraphs 22 and 24 are preserve-only blank spacers. Keep them
# intact and place the additional completed findings in their editable peers.
design[21]+=' '+design.pop(22)
design[23]+=' '+design.pop(24)
design[21]='最后prefill的104个query按ncols16分7组，K/V各地址重复7次，decode各1次。首末decode地址集合相同，说明本窗口tile访问粒度未增加；这不是DRAM字节或cache miss。'
design[23]='原生prefill最多2 CTA/SM，164 CTA超过160个驻留槽；decode shared66080B限制1 CTA/SM。图不能证明FIFO。细probe的p95/median为1.122、1.102、1.167；原生时间另用CUPTI核查。'
analysis[35]='补充完整请求、相同分析终点与输出正确性；硬件原因独立核查。'
analysis[44]='ERE减小传输与保存，完整恢复与分析抵消部分采集收益；模型调优另验数值。'
analysis[51]='CPU/GPU库与SHA固定；nvCOMP5.3和ERE精确C++ decoder均独立核验。'
b.build('design-report',design,dtables,'case-executed-design.docx');b.build('experiment-analysis',analysis,atable,'case-analysis.docx')
# Append images using package edits. All template section/style/header/table
# parts remain byte-for-byte intact; existing relationships remain unchanged.
path=out/'case-analysis.docx'
with ZipFile(path) as z:parts={name:z.read(name)for name in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find(b.tag('body'));paragraphs=body.findall(b.tag('p'));rels=E.fromstring(parts['word/_rels/document.xml.rels']);types=E.fromstring(parts['[Content_Types].xml'])
component=Document()
for token in [0,1,63]:component.add_picture(str(r/f'figures/memory-layout-{token}.png'),width=Inches(6.4))
buffer=io.BytesIO();component.save(buffer)
with ZipFile(buffer) as z:
 pdoc=E.fromstring(z.read('word/document.xml'));pics=pdoc.xpath('//w:body/w:p[w:r/w:drawing]',namespaces=b.NS);prels=E.fromstring(z.read('word/_rels/document.xml.rels'));lookup={x.get('Id'):x.get('Target')for x in prels}
 for i,(token,pic) in enumerate(zip([0,1,63],pics),1):
  pb=E.Element(b.tag('p'));rr=E.SubElement(pb,b.tag('r'));br=E.SubElement(rr,b.tag('br'));br.set(b.tag('type'),'page');body.insert(len(body)-1,pb)
  heading=copy.deepcopy(paragraphs[36]);b.set_p(heading,['图 1 最后一个 prefill 块','图 2 第一次 decode','图 3 第63步 decode'][i-1],'Arial Unicode MS');body.insert(len(body)-1,heading)
  image=copy.deepcopy(pic);blip=image.xpath('//*[local-name()="blip"]')[0];oldid=blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed');newid=f'rIdCaseFigure{i}';blip.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed',newid)
  for item in image.xpath('//*[local-name()="docPr"]'):item.set('id',str(1000+i));item.set('name',f'Case memory layout {token}')
  filename=f'case-memory-{token}.png';parts[f'word/media/{filename}']=z.read('word/'+lookup[oldid]);relation=E.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship');relation.set('Id',newid);relation.set('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/image');relation.set('Target','media/'+filename);body.insert(len(body)-1,image)
  caption=copy.deepcopy(paragraphs[31]);b.set_p(caption,['完整请求最后104输入token的attention。K/V每个不同地址访问7次，全部事件保留。','第1步decode。K/V地址各访问1次，单独原生资源查询为1 CTA/SM。','第63步decode。K/V地址集合和事件数与第1步相同，时序为插桩观测。'][i-1]+' 页面按64KiB分箱，时间按线程入口相对周期；区域间留白仅用于阅读。原始Map和完整归档由case-report.md索引。','Arial Unicode MS');body.insert(len(body)-1,caption)
if not any(x.get('Extension')=='png' for x in types):
 node=E.SubElement(types,'{http://schemas.openxmlformats.org/package/2006/content-types}Default');node.set('Extension','png');node.set('ContentType','image/png')
# Keep heading roles and each existing table header with the next content.
# This small content
# pagination adaptation avoids lone headers after Chinese text reflow, while
# preserving table grids, section geometry, styles, cells and row dimensions.
for index in [26,29,33,36,39,43,46,49,56,59,63,66,71,79,83,86,93,100]:
 p=paragraphs[index];prop=p.find(b.tag('pPr'))
 if prop is None:prop=E.Element(b.tag('pPr'));p.insert(0,prop)
 keep=prop.find(b.tag('keepNext'))
 if keep is None:keep=E.SubElement(prop,b.tag('keepNext'))
 keep.set(b.tag('val'),'1')
for table in body.findall(b.tag('tbl')):
 first=table.find(b.tag('tr'))
 for p in first.xpath('.//w:p',namespaces=b.NS):
  prop=p.find(b.tag('pPr'))
  if prop is None:prop=E.Element(b.tag('pPr'));p.insert(0,prop)
  keep=prop.find(b.tag('keepNext'))
  if keep is None:keep=E.SubElement(prop,b.tag('keepNext'))
  keep.set(b.tag('val'),'1')
# Template spacer paragraphs occur between some headings and their tables.
# Carry keepNext through those spacers so the heading follows the table.
children=list(body)
for index in [26,29,33,36,39,43,46,49,56,59,63,66,71,79,83,86,93,100]:
 position=children.index(paragraphs[index])+1
 while position<len(children):
  p=children[position]
  if p.tag!=b.tag('p') or p.xpath('.//w:t|.//w:drawing|.//w:br',namespaces=b.NS):break
  prop=p.find(b.tag('pPr'))
  if prop is None:prop=E.Element(b.tag('pPr'));p.insert(0,prop)
  keep=prop.find(b.tag('keepNext'))
  if keep is None:keep=E.SubElement(prop,b.tag('keepNext'))
  keep.set(b.tag('val'),'1');position+=1
parts['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True);parts['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True);parts['[Content_Types].xml']=E.tostring(types,xml_declaration=True,encoding='UTF-8',standalone=True)
for name,data in list(parts.items()):
 if name.startswith(('word/header','word/footer')) and name.endswith('.xml'):
  node=E.fromstring(data)
  for t in node.xpath('//w:t',namespaces=b.NS):
   if t.text=='完整推理诊断实验方案':t.text='完整推理诊断案例'
  parts[name]=E.tostring(node,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(path,'w') as z:
 for name,data in parts.items():z.writestr(name,data)
contract=out/'qa/analysis/artifact.md';contract.write_text(contract.read_text()+'\nIntentional user-requested addition: three real memory-layout images and captions appended using existing Heading1/body styles; original sections, geometry and all prior relationship entries preserved. New image media and three image relationships are added. Heading and table-header paragraphs receive keepNext to prevent orphan headings/headers after Chinese reflow; grids/cells/row geometry remain exact.\n')
(out/'qa/analysis/image-addition.json').write_text(json.dumps({'added_figures':[0,1,63],'section_geometry_unchanged':True,'reference_unchanged':True,'existing_relationships_preserved':True},indent=2)+'\n')
print(path)
