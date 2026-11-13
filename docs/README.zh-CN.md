# ERETrace 中文指南

**ERETrace** 是已被 ISPA 2026 录用的论文 *Execution-Reference Encoding and Recovery-Aware Collection for GPU Profiling* 的公开研究实现。
通过 SM/CTA 执行域参考、受检窄整数编码及有界异步采集，在声明的捕获范围内逐条保留原始字段、精度、顺序和归属；编码无法接受时恢复受保留的 raw 数据，无需重放 kernel。

## 安装与验证

```bash
git clone https://github.com/DQR-Happy/ERETrace.git
cd ERETrace
python3 -m venv .venv
source .venv/bin/activate
python scripts/bootstrap.py
python -m pip install -e '.[test]'
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/check_repository.py
python scripts/summarize_paper.py
```

Bootstrap 从上游直接获取固定版本 Neutrino，应用公开补丁，逐文件验证 SHA-256。Neutrino 的 Python 命名空间和环境变量保持兼容。模型、驱动及 vendor SDK 不会被隐式下载或替换。

GPU 采集需 Linux + CUDA，安装和 llama.cpp 固定版本步骤见 [INSTALL.md](INSTALL.md)。论文主实验使用 RTX 3090；Jetson 发布冒烟检查与原论文性能数据分别记录在 [VALIDATION.md](VALIDATION.md)。下载权重优先使用 ModelScope，并核对 [models.json](../configs/models.json) 的文件哈希。

## 功能与复现

- [完整推理主实验及消融](REPRODUCE.md)：`native serial async nvcomp ours sm24 sm32`，包括 B1–B16 和长度敏感性。
- [真实文档问答](DOCUMENT_QA.md)：实际文本输入、完整推理捕获、阶段归因和布局诊断。
- [NVBit 与 XTrace 适配](FRONTENDS.md)：分别注明 SASS 和 PTX replacement 路径，相关 SDK/runtime 独立安装。
- [完整实验地图](EXPERIMENTS.md)：源码、依赖、执行入口和实验范围。

主时间从完整推理请求开始到完整 trace 文件关闭，包含探针、同步、编码检查/raw 恢复、D2H、drain 和输出。离线解码/分析及额外 oracle/SHA 校验不纳入正式性能计时。论文数据为已保存测量值，发布测试不是重新测量全部论文结果。

## 引用与许可

作者：Qirui Deng、Yongxian Fan、Yongwei Zhang、Zuwen Lai，见 [CITATION.cff](../CITATION.cff)。引用信息见 [CITATION.md](CITATION.md)。
自研新增代码、补丁、文档和原创视觉资产采用 MIT。上游及 vendor 组件保持各自归属和条款，详见 [第三方声明](../THIRD_PARTY_NOTICES.md)。
