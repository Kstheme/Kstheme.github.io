---
title: 关于
---

# 余弦

让机器看见文档、读懂内容、理解意义。

过去三年，我一直专注于**计算机视觉与大模型融合**的工程落地——教会 AI 处理真实世界中那些不完美的文档：模糊的、畸变的、有阴影的、手写的、劣质印刷的。然后提取它们的结构，用大模型去理解、分析和解答。

交付的不是 Demo，是每天处理成千上万页、在真实业务中稳定运行的线上系统。

---

## 我做什么

从零开始设计并交付端到端 AI 系统。覆盖完整链路：图像预处理 → 版面分析 → OCR 结构化还原 → 大模型判评 → RAG 检索 → 知识图谱推理。架构、模型训练、部署优化、全链路打通——全部自己负责。

**核心深耕领域：**

- **文档智能** — OCR 管线设计、版面分析、真实场景下退化文档的展平与增强
- **大模型后训练与对齐** — LoRA/SFT 微调、指令样本构造、结构化输出对齐、大规模 Prompt 工程
- **RAG 与知识图谱** — GraphRAG 架构、混合检索（BM25 + 向量 + 图谱）、Neo4j 建模、多跳推理
- **Agent 系统** — LangGraph 多智能体工作流、任务规划、工具调用、记忆机制、生产级编排
- **生产工程** — ONNX/TensorRT 加速、vLLM 推理部署、FastAPI 服务、Paddle-Lite 端侧部署

---

## 交付过的系统

<VPCardGrid :cols="{ sm: 1, md: 2, lg: 2 }">

<VPCard title="智能作业批改系统" icon="material-symbols:auto-detect">

从拍照上传到批改出分，全自动流水线。处理模糊的教室拍摄、卷曲的试卷、混合手写印刷的内容。**1000 样本准确率 97%，单页 1–1.5 秒。**

<Badge type="tip" text="RT-DETR" />
<Badge type="tip" text="U2Net" />
<Badge type="tip" text="OCR" />
<Badge type="tip" text="ONNX/TensorRT" />

</VPCard>

<VPCard title="智学作文智能判评平台" icon="material-symbols:edit-document">

FAST-AND-SLOW 双引擎架构：微调后的 Qwen2.5-32B 负责深度推理，轻量 MLP 快速处理常规作文。已上线 APP、公众号、Web 端，服务真实用户。**OCR 准确率 98.5%，与人工评分吻合率 ~90%。**

<Badge type="tip" text="Qwen2.5-32B" />
<Badge type="tip" text="LoRA" />
<Badge type="tip" text="vLLM" />
<Badge type="tip" text="LangGPT" />

</VPCard>

<VPCard title="K12 数学知识图谱问答" icon="material-symbols:account-tree">

基于 GraphRAG 的数学智能问答系统。Neo4j + Milvus 双擎架构，混合检索自动路由——简单问题走向量搜索，复杂推理触达图谱多跳查询。支持知识点讲解、例题解析、前置诊断。

<Badge type="tip" text="Neo4j" />
<Badge type="tip" text="Milvus" />
<Badge type="tip" text="GraphRAG" />
<Badge type="tip" text="DeepSeek" />

</VPCard>

<VPCard title="Study Planner AI Agent" icon="material-symbols:calendar-clock">

独立设计、开发并上线的 AI 学习规划助手。六阶段 LangGraph Agent 工作流，Planner-Critic 修复循环持续优化学习计划。完整 RAG 管线 + 进度追踪 + 可视化看板。**30+ 测试用例，双模式架构。**

<Badge type="tip" text="LangGraph" />
<Badge type="tip" text="Agent" />
<Badge type="tip" text="RAG" />
<Badge type="tip" text="Streamlit" />

</VPCard>

</VPCardGrid>

---

## 我的工作方式

我相信最好的 AI 系统是由能看见全貌的人建造的——从像素到 Prompt 到生产端点。这就是我的工作方式：不切割分工，而是深扎全栈。

我也相信 AI 辅助开发是新常态。我实践的 **Spec-Driven AI Development**——先写 Spec 和测试契约，再用 AI 工具迭代直到测试通过。快速、可靠、从第一天就面向交付。

---

## 一起做点什么

如果你有文档与智能结合的难题——合同分析、表单处理、知识库自动化、或者某个我还未曾想到的方向——我很乐意聊聊。

`killkstheme@gmail.com` · [GitHub](https://github.com/Kstheme) · [知乎](https://www.zhihu.com/people/kstheme)

中国图象图形学学会（CSIG）会员
