<div align="center">
  <img src="./static/logo/logo.png" width="360" alt="Z-Eval Logo" />

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](./LICENSE)
[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2603.09821)

</div>

  <h4 align="center">
    <i>Z-Eval，一句话完成模型评测</i>
  </h4>
  <br>

> **Z-Eval** 基于 [OpenDCAI/One-Eval](https://github.com/OpenDCAI/One-Eval) 开源项目。经过两周的密集开发，我们在原始代码基础上进行了大量增强，包括多模型竞对评测、LLM Judge 评判系统、本地数据集上传、任务持久化以及全面的前端重构。

[English](./README.md) | 简体中文

<p align="center">
  <img src="https://github.com/user-attachments/assets/129d9826-a48b-4ab5-8006-ad7d9595cc95" alt="中文版演示" width="70%">
</p>

## 📰 1. 最新动态

- **[2026-03] Z-Eval v1.0.0 发布！**
  基于 OpenDCAI/One-Eval，新增多模型并行评测、LLM Judge 评判系统、本地数据集上传、任务持久化、结果预览与下载，以及全面的前端 UI 改版。详见下方 [更新日志](#-2-z-eval-新增了什么)。

## 🔍 2. Z-Eval 新增了什么？

Z-Eval 继承了 [OpenDCAI/One-Eval](https://github.com/OpenDCAI/One-Eval) 强大的 NL2Eval 引擎，并在此基础上进行了大量扩展：

### 🔀 多模型竞对评测
- 在设置页面注册多个 API 模型（兼容 OpenAI、Claude、Gemini 等）。
- 选择任意模型组合，在同一基准上并行评测，直观对比模型表现。
- 提供跨模型的评分对比视图和详细分数分解。

### 🤖 LLM Judge 评判系统
- 内置 LLM Judge 面板，利用配置的模型对开放式或主观性回答进行评判。
- 支持自定义评判 Prompt 和可配置的 Judge 模型选择。
- 评判分数与自动化指标一起整合到评测摘要中。

### 📁 本地数据集上传
- 支持从前端直接上传自定义评测数据集（JSON/JSONL/CSV 格式）。
- 上传的数据集存储在服务端，可在评测任务中像内置基准一样选择使用。

### 💾 任务持久化与历史记录
- 评测任务自动持久化到磁盘，不再因刷新页面丢失进度。
- 支持恢复中断的任务，或直接查看历史评测结果，无需重新运行。
- 任务历史记录可在工作台侧边栏中访问。

### 📊 结果预览与下载
- 在下载前可通过结构化弹窗预览评测结果。
- 支持多种格式导出（JSON、CSV），方便后续分析和报告生成。
- 摘要面板支持一键预览和批量下载。

### ⚙️ 增强的模型管理
- 注册模型的完整增删改查：添加、编辑、删除、启用/停用。
- "全选"按钮支持快速批量操作。
- 编辑模式下 API Key 自动隐藏，保障安全。
- 支持自定义模型名称，灵活标识不同模型。

### 🎨 UI/UX 全面改版
- 前端界面全面视觉焕新，采用现代化简洁设计风格。
- 改进多模型选择交互界面，操作更加直观。
- 优化 Tab 布局，提升评测工作台的导航体验。

## 💡 为什么选择 Z-Eval？

以往的模型评测框架通常需要用户自行寻找 Benchmarks、下载数据，并手动填写大量配置参数。
**Z-Eval** 旨在改变这一现状：**凡是能自动做到的，都将交给 Agent 自动完成**。结合多模型竞对、LLM Judge 和本地数据集支持，我们提供了一站式评测平台——从基准发现到细粒度指标分析，全面覆盖。

## 🔍 3. 项目概览

Z-Eval 将评测重构为**图化执行过程 (Graph / Node / State)**，基于 [DataFlow](https://github.com/OpenDCAI/DataFlow) 与 [LangGraph](https://github.com/langchain-ai/langgraph) 构建：

- 🗣️ **NL2Eval**: 只需输入一段自然语言目标（例如"评估模型在数学推理任务上的表现"），系统自动解析意图并规划执行路径。
- 🧩 **全链路自动化**: 自动完成基准推荐、数据准备、推理执行、指标匹配、打分与多维度报告生成。
- ⏸️ **人机交互**: 支持关键节点（如基准选择、结果复核）的中断与人工干预，便于根据反馈实时调整评测策略。
- 📊 **可扩展架构**: 基于 DataFlow 的算子体系与 LangGraph 的状态管理，轻松集成私有数据集与自定义指标。

![Z-Eval Framework](./static/logo/eval_framework.png)

## ⚡ 4. 快速开始

### 4.1 安装环境（推荐方式）

提供了 Conda 与 uv 两种环境管理方式，任选其一即可快速上手：

#### 方式 A: Conda
```bash
conda create -n one-eval python=3.11 -y
conda activate one-eval
pip install -e .
```

#### 方式 B: uv
```bash
uv venv
uv pip install -e .
```

### 4.2 启动服务

Z-Eval 采用前后端分离架构，请分别启动后端 API 与前端界面。

#### ① 启动后端 (FastAPI)
```bash
uvicorn one_eval.server.app:app --host 0.0.0.0 --port 8000
```

#### ② 启动前端 (Vite + React)
```bash
cd one-eval-web
npm install
npm run dev
```

访问 http://localhost:5173 即可开始交互式评测。

> 启动后应先进入设置界面，配置 API、模型以及 HF Token 等参数（以支持批量下载数据），并点击保存。

### 4.3 极简代码模式（开发者模式）

如果你更喜欢在代码中直接调用，可以直接运行内置的完整工作流示例：
[workflow_all.py](./one_eval/graph/workflow_all.py)

```bash
# 示例：直接通过命令行发起一次数学能力评估
python -m one_eval.graph.workflow_all "我想评估我的模型在Reasoning上的表现"
```

该 Graph 展示了从 Query 解析到报告生成的完整闭环，欢迎基于此进行二次开发与节点扩展。

## 🗂️ 5. 评测基准库 (Bench Gallery)

Z-Eval 内置了丰富的 **Bench Gallery**，用于统一管理各类评测基准的元信息（如任务类型、数据格式、Prompt 模板）。

> 目前已涵盖主流纯文本能力维度（无需复杂沙盒环境）：
> - 🧮 **Reasoning**: MATH, GSM8K, BBH, AIME...
> - 🌐 **General Knowledge**: MMLU, CEval, CMMLU...
> - 🔧 **Instruction Following**: IFEval...

![Bench Gallery](./static/logo/gallery.png)

## 🚀 6. 未来规划

我们计划在未来继续维护并从以下方向更新 Z-Eval：

- 💻 **支持复杂评测场景**: 扩展对 Code、Text2SQL 等需要额外执行环境的 LLM 评测领域的支持。
- 🤖 **Agentic 评测与沙盒环境**: 支持基于复杂沙盒环境的 Agentic 领域评测（如 SWE-bench 等）。
- 📈 **增强 LLM Judge**: 更多评判策略、思维链评判、多 Judge 集成，提升主观评测的可靠性。
- 🌐 **在线社区与平台部署**: 部署在线评测平台，支持用户讨论交流、构建私有 Benchmark，并实现共享与复用。

## 📮 7. 联系方式

如果您对本项目感兴趣，或有任何疑问与建议，欢迎通过 Issue 联系我们。

- 📮 [GitHub Issues](../../issues)：提交 Bug 或功能建议。
- 🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进。

## 致谢

Z-Eval 基于 [OpenDCAI/One-Eval](https://github.com/OpenDCAI/One-Eval) 的优秀工作构建。我们衷心感谢原作者对开源社区的贡献。

## Citation

```bibtex
@misc{shen2026oneevalagenticautomatedtraceable,
      title={One-Eval: An Agentic System for Automated and Traceable LLM Evaluation},
      author={Chengyu Shen and Yanheng Hou and Minghui Pan and Runming He and Zhen Hao Wong and Meiyi Qiang and Zhou Liu and Hao Liang and Peichao Lai and Zeang Sheng and Wentao Zhang},
      year={2026},
      eprint={2603.09821},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.09821},
}

@article{liang2025dataflow,
  title={DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI},
  author={Liang, Hao and Ma, Xiaochen and Liu, Zhou and Wong, Zhen Hao and Zhao, Zhengyang and Meng, Zimo and He, Runming and Shen, Chengyu and Cai, Qifeng and Han, Zhaoyang and others},
  journal={arXiv preprint arXiv:2512.16676},
  year={2025}
}
```

## License

本项目基于 Apache License 2.0 开源 — 详见 [LICENSE](./LICENSE) 文件。
