---
title: 关于
---

# 余弦

让机器看见文档、读懂内容、理解意义。

过去三年，我一直专注于**计算机视觉与大模型融合**的工程落地——教会 AI 处理真实世界中那些不完美的文档：模糊的、畸变的、有阴影的、手写的、劣质印刷的。然后提取它们的结构，用大模型去理解、分析和解答。

交付的不是 Demo，是每天处理成千上万页、在真实业务中稳定运行的线上系统。

现在我在西交利物浦大学读**模式识别与智能系统**硕士，把工程直觉往上游推一步——关注 AI4Math、LLM Reasoning 与高效推理。

---

## 我做什么

从零开始设计并交付端到端 AI 系统。覆盖完整链路：图像预处理 → 版面分析 → OCR 结构化还原 → 大模型判评 → RAG 检索 → 知识图谱推理。架构、模型训练、部署优化、全链路打通——全部自己负责。

**核心深耕领域：**

- **文档智能** — OCR 管线设计、版面分析、真实场景下退化文档的展平与增强
- **大模型后训练与对齐** — LoRA/SFT 微调、指令样本构造、结构化输出对齐、大规模 Prompt 工程
- **RAG 与知识图谱** — GraphRAG 架构、混合检索（BM25 + 向量 + 图谱）、Neo4j 建模、多跳推理
- **Agent 系统** — LangGraph 多智能体工作流、任务规划、工具调用、记忆机制、生产级编排
- **大模型推理与 AI4Math** — Chain-of-Thought、Self-Consistency、Test-Time Scaling、Verifier / Reward Model
- **生产工程** — ONNX/TensorRT 加速、vLLM 推理部署、FastAPI 服务、Paddle-Lite 端侧部署

---

## 教育背景

**西交利物浦大学** | 模式识别与智能系统，人工智能与先进技术学院 | 硕士研究生 —— 2026.09 - 至今

主修课程：模式识别、强化学习、自然语言处理、语音语言处理

**宁波工程学院** | 计算机科学与技术，网络空间安全学院 | 工学学士 —— 2018.09 - 2022.06（GPA 3.9/4）

主修课程：微积分、C/C++、Java、Python、数据结构、计算机组成原理、操作系统、计算机网络、机器学习、自然语言处理、数字图像处理

---

## 工作经历

**北京南昊科技股份有限公司** | 技术经理 | AI 算法研发 —— 2022.07 - 2025.10

主导文档智能与 OCR + LLM 系统的算法研发与落地，覆盖扫描文档、手写文本、复杂版面等真实场景。

- **系统架构设计** — 主导多个 CV + OCR + LLM 系统的整体方案设计，打通从图像采集、预处理、识别理解到结果生成的完整链路
- **模型训练与优化** — 基于 PyTorch、PaddlePaddle、ModelScope 完成模型训练、调优与推理优化，重点解决畸变、阴影、噪声、模糊、小样本等真实场景问题
- **大模型后训练与部署** — 负责判评大模型业务数据 SFT 样本构造、LoRA 微调、结构化 Prompt 设计与 vLLM 推理部署与加速优化
- **团队协同** — 指导标注团队与算法团队推进数据标注、训练优化与上线验证，并与业务团队持续对齐标准与需求

---

## 交付过的系统

<VPCardGrid :cols="{ sm: 1, md: 2, lg: 2 }">

<VPCard title="智能作业批改系统" icon="material-symbols:auto-detect">

面向扫描文档场景的智能识别与自动判评系统，打通「图像采集 → 预处理 → 版面分析 → OCR 识别 → 结构化重建 → 判评生成」完整链路。基于 RT-DETR 做区域检测并结合匈牙利算法完成模板匹配；设计差异化多路 OCR 路由（中文/英文/公式），推动整体识别准确率提升约 12%。**1000 张测试样本整体准确率 97%，单文档响应 1–1.5 秒。**

<Badge type="tip" text="RT-DETR" />
<Badge type="tip" text="U2Net" />
<Badge type="tip" text="DewarpNet" />
<Badge type="tip" text="OCR" />
<Badge type="tip" text="ONNX/TensorRT" />

</VPCard>

<VPCard title="智学作文智能判评平台" icon="material-symbols:edit-document">

FAST-AND-SLOW 双引擎架构：Slow 模块用微调后的 Qwen2.5-32B 深度推理生成结构化评估报告，Fast 模块在 LLM 表征后接全连接层做快速分数预测，并按置信度阈值决定是否触发重评估。基于 Swift 对 Qwen2.5-32B 做 LoRA 微调（2000 条多难度标注样本）。已上线 APP、公众号、Web 端。**OCR 准确率 98.5%，与人工评分误差 ≤5 分的比例约 90%。**

<Badge type="tip" text="Qwen2.5-32B" />
<Badge type="tip" text="Swift/LoRA" />
<Badge type="tip" text="vLLM" />
<Badge type="tip" text="LangGPT" />

</VPCard>

<VPCard title="K12 数学知识图谱问答" icon="material-symbols:account-tree">

基于 GraphRAG 的数学智能问答系统。Neo4j + Milvus 双擎架构，将知识点、公式、题型、解题步骤、易错点与前置依赖建模为图结构；实现 BM25 + 向量 + 图索引的多路召回并用 RRF 融合排序。设计 LLM 驱动的查询路由，按问题复杂度自动选择混合检索、GraphRAG 检索或组合策略。

<Badge type="tip" text="Neo4j" />
<Badge type="tip" text="Milvus" />
<Badge type="tip" text="GraphRAG" />
<Badge type="tip" text="RRF" />
<Badge type="tip" text="DeepSeek" />

</VPCard>

<VPCard title="Study Planner AI Agent" icon="material-symbols:calendar-clock">

独立设计、开发并上线的 AI 学习规划助手。六阶段 LangGraph Agent 工作流（Profile → Knowledge → Resource → Planner → Critic → Output），Planner-Critic 修复循环持续优化学习计划。完整 RAG 管线 + 进度追踪 + 可视化看板。**30+ pytest 用例，Fake/Real 双模式架构。**

<Badge type="tip" text="LangGraph" />
<Badge type="tip" text="Agent" />
<Badge type="tip" text="RAG" />
<Badge type="tip" text="PostgreSQL" />
<Badge type="tip" text="Streamlit" />

</VPCard>

<VPCard title="细粒度检测与实例分割系统" icon="material-symbols:polyline">

针对手工场景中细长目标的自动检测需求，构建高鲁棒性检测与匹配系统。基于 YOLOv8-Seg 做实例分割解决细长目标易漏检问题，融合霍夫直线检测施加几何约束以提高端点定位精度。完成 ONNX 转换与 TensorRT 部署优化。**真实场景准确率超 95%。**

<Badge type="tip" text="YOLOv8-Seg" />
<Badge type="tip" text="OpenCV" />
<Badge type="tip" text="霍夫变换" />
<Badge type="tip" text="TensorRT" />

</VPCard>

<VPCard title="答题卡涂点阵列自动识别" icon="material-symbols:check-box-outline">

面向无人值守场景的答题卡自动检测、校正与涂点识别系统。设计并优化文档切割与矫正算法提升拍照场景对齐精度，实现涂点区域自动识别与判定逻辑。全部识别模型端侧部署，降低网络依赖、提升实时响应。

<Badge type="tip" text="Paddle-Lite" />
<Badge type="tip" text="YOLOv8" />
<Badge type="tip" text="端侧部署" />

</VPCard>

</VPCardGrid>

### 其他项目

- **白纸试卷自适应切边矫正** — 基于轮廓检测、四点透视变换与 OTSU 二值化实现 A3/A4 混合纸张自适应裁剪，解决白底干扰与边界模糊问题，切割成功率超 **98%**
- **扫描阅卷手写考号识别** — 基于 CRNN + CTC Loss 训练手写数字识别模型，ONNX 导出 + C++ 离线推理部署，真实数据集识别率约 **99%**
- **中文作文字词句智能批改** — 基于 LLM + LangChain 构建轻量级文本纠错引擎，通过编辑距离实现字符级错误定位，支持结构化 JSON 输出
- **高考作文主题预测** — 基于 BERTopic + BGE embedding + UMAP + HDBSCAN 构建主题建模系统，实现作文热点自动挖掘与关键词提取
- **AI 教案生成** — 基于 Prompt Engineering + LangGPT + LangChain 构建教案生成应用，支持多学科模板生成与内容结构化输出

---

## 开源贡献与技术影响力

- **技术内容创作** — 维护个人技术博客 [kstheme.github.io/blog](https://kstheme.github.io/blog/)，持续发布 LLM、Agent、RAG、AI4Math、计算机视觉与 AI 工程实践相关技术文章；同步运营微信公众号 **Kstheme for AI**，目前关注用户 250+
- **Nature Skills Contributor** — 向 Nature Skills 提交并被接受开源 PR，贡献 **Nature Paper Card** Skill，用于论文精读、关键信息提取与结构化总结，帮助研究者快速理解论文的研究问题、核心方法、实验结果与局限性
- **Open Source Collaboration** — 参与开源社区协作，包括向 psmux 提交 Issue，对实际使用中发现的问题进行复现、描述与改进建议反馈

---

## 我的工作方式

我相信最好的 AI 系统是由能看见全貌的人建造的——从像素到 Prompt 到生产端点。这就是我的工作方式：不切割分工，而是深扎全栈。

我也相信 AI 辅助开发是新常态。我实践的 **Spec-Driven AI Development**——先写 Spec 和测试契约，再用 AI 工具迭代直到测试通过。快速、可靠、从第一天就面向交付。

---

## 一起做点什么

如果你有文档与智能结合的难题——合同分析、表单处理、知识库自动化、或者某个我还未曾想到的方向——我很乐意聊聊。

`killkstheme@outlook.com` · [GitHub](https://github.com/Kstheme) · [知乎](https://www.zhihu.com/people/kstheme)

中国图象图形学学会（CSIG）会员（2023 年入会）· PTE 69（等价 IELTS 7.0）
