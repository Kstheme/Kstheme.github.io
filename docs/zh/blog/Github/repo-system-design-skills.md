---
author: Kstheme
date: 2025-11-10T00:00:00.000Z
category:
  - 开源项目
tags:
  - system-design
  - github
  - agent
title: "把任何 GitHub 仓库变成系统设计课：这个开源项目做到了"
createTime: 2026/06/21 23:26:20
permalink: /zh/article/repo-system-design-skills/
---

> 你是不是也这样：接手一个项目先跑起来再说，遇到 Bug 就加日志，看到新 PR 就 merge，但一年下来问你这个项目架构怎么设计的，你答不上来？
>
> 不只是你。**大多数程序员对系统设计的理解，是"用过"而不是"学过"。**

---

## 01 一个问题：为什么学了那么多，系统设计还是没入门？

很多人学系统设计的方法是：

- 刷《系统设计面试》；
- 背几个经典案例（设计 Twitter、设计 Uber）；
- 面试前看几篇架构博客。

结果呢？

面试官问"你当前项目里用的是什么消息队列，为什么选它？"，你只能说"别人选的"。

**真正的问题不是你没有系统设计知识，而是你从来没有用源码证据去理解过一个真实系统的架构。**

你写的代码、你改的 Bug、你加的 Feature，都是系统设计的素材——但你缺一个帮你把它们串起来的方法。

---

## 02 这个项目是什么？一句话讲清楚

这个项目叫 **Repo System Design Skills**，是一套给 AI 编程助手（Codex / Claude Code）用的 Skills。

它的目标不是让 Agent 帮你写代码，而是让 Agent 成为一个**源码证据驱动的系统设计导师**。

![](/images/repo-system-design/skill-overview.png)

> **一句话：它把任意一个 GitHub 仓库，变成一门系统设计课。**

---

## 03 它解决了什么老问题？

我们先看看在没有它之前，用 Agent 学项目是什么体验。

### 你可能会这样问 Agent

```
帮我看一下这个项目的架构。
```

Agent 会给你一段看起来很有道理的话：

- "本项目采用分层架构……"
- "前端使用 React……"
- "后端使用 Spring Boot……"

听着都对，但**它没有引用任何源码**。你不知道它是在读代码还是在编。

### 更糟糕的是

| 问题         | 普通 Agent 回答 | 这个 Skill 的回答                         |
| ------------ | --------------- | ----------------------------------------- |
| 架构怎么分层 | 模板化回答      | 引用具体的文件和目录                      |
| 核心链路     | 泛泛概括        | 从入口追到数据库，每一步有源码            |
| 设计取舍     | 只给结论        | 区分 Evidence / Inference / Open Question |
| 优化建议     | 可能推倒重来    | 先出验证方案，用 branch 安全实验          |
| 学习效果     | 听完就忘        | 反向出题、纠错、复习卡                    |

**这个项目的核心创新不是"帮你看代码"，而是建立了 Agent 教你学系统设计的完整方法论。**

---

## 04 五个 Skill，一条完整学习闭环

项目包含 5 个 Skill，正好对应系统设计学习的 5 个层次。

```
                   ┌──────────────────┐
                   │  repo-system-    │
                   │  design-lab      │  ← 总控：规划学习路线
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  repo-           │
                   │  architecture-   │  ← 架构图：模块边界、依赖、数据流
                   │  mapper          │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  repo-flow-      │
                   │  tracer          │  ← 链路追踪：从 UI 到 DB 追溯完整路径
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  repo-design-    │
                   │  optimizer       │  ← 优化评审：找瓶颈、设计实验
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  repo-design-    │
                   │  coach           │  ← 教练模式：反向提问、面试训练
                   └──────────────────┘
```

### Skill 1：repo-system-design-lab（总控实验室）

这是入口。它不直接做分析，而是制定学习计划。

你告诉它"我想学这个项目"，它会：

1. 先读 README、配置、目录结构，输出**项目定位**；
2. 找到运行入口、核心模块、外部依赖；
3. 推荐**最值得深入学习的 3-5 条核心链路**；
4. 在你确认之前，**绝不改一行代码**。

### Skill 2：repo-architecture-mapper（架构地图绘制）

这个 Skill 会把你的项目"画"出来。

- 分层架构图
- 运行时 / 部署图
- 数据所有权图
- 依赖图
- Fake / Real 模式切换图

**每个图都有源码引用证据。** 不只是画个框，而是告诉你"这个模块对应的代码文件夹是 src/services/，入口是 main.go:42"。

### Skill 3：repo-flow-tracer（链路追踪）

这是我觉得最实用的一个。

假设你想理解"用户登录 -> 鉴权 -> Token 颁发"这条链路：

```
Streamlit 页面 (app.py:120)
  → 调用 LoginUseCase (application/auth.py:45)
    → UserRepository 查询用户 (domain/models.py:200)
      → Postgres 或 InMemory 适配器 (infrastructure/repositories.py:88)
        → JWT Token 生成 (infrastructure/auth.py:30)
          → 返回结果给页面
```

**每一步都告诉你文件位置和行号。** 你可以在编辑器里对照着学。

### Skill 4：repo-design-optimizer（设计优化师）

当你想优化项目时，这个 Skill 会：

1. 从 **6 个维度**（性能、可靠性、扩展性、可维护性、可观测性、安全性）全面扫描；
2. 每个问题都附上**源码证据、影响程度、优先级、优化方案**；
3. **先设计验证方案，再改代码**；
4. 自动建议 branch 或 worktree 方案，做好回滚准备。

真正改代码前要有验证计划，这个习惯能救你很多次。

### Skill 5：repo-design-coach（学习教练）

这是最特别的一个——它反过来考你。

- 一次只问一个问题；
- 你回答后，它用**源码证据纠正你**；
- 从 Level 1 到 Level 5 逐步加深；
- 帮你形成**面试式的表达方式**。

---

## 05 技术架构：这个项目是怎么工作的？

这个项目本身也值得学习。它面向的不是"人类读者"，而是 **AI Agent**。

```
repo-system-design-skills/
  codex/
    skills/                 # Codex 格式，包含 agents/openai.yaml
      repo-system-design-lab/
      repo-architecture-mapper/
      repo-flow-tracer/
      repo-design-optimizer/
      repo-design-coach/
  claude/
    skills/                 # Claude Code 格式，SKILL.md + references
      repo-system-design-lab/
      repo-architecture-mapper/
      repo-flow-tracer/
      repo-design-optimizer/
      repo-design-coach/
```

### 值得注意的设计亮点

**亮点 1：双平台兼容**

同一套方法论同时兼容 Codex（OpenAI）和 Claude Code。两个版本的 Skill 逻辑相同，只是格式不同（Codex 需要 agents/openai.yaml 元数据，Claude 用 SKILL.md）。

**亮点 2：Skill 分层设计**

采用"总控 + 专用执行 + 反向教学"三层，每层用途明确：

- 总控负责编排学习流程；
- 专用 Skill 负责单领域深度分析；
- 教练 Skill 验证学习效果。

**亮点 3：引用驱动的证据链**

每条架构结论都必须引用源码文件。这是这套 Skill 最有价值的设计——它强迫 Agent 读代码而不是编结论。

**亮点 4：区分 Evidence / Inference / Open Question**

这是最值得面试时讲的设计细节。不是所有结论都能从源码确定，区分这三种状态才是系统设计的成熟思维。

---

## 06 核心难点：做一个让 Agent 教人的 Skill 难在哪？

### 难点 1：Agent 很容易编造

Agent 的天性是生成看起来合理的文本，而不是严谨推理。

这个项目的解法是：**强制要求每个架构结论都引用源码文件和行号。** 如果 Agent 不能引用源码，就不算 Evidence，只能算 Inference。

### 难点 2：Agent 很难把握"只读"边界

Agent 被问到项目问题时总想写代码。

这个项目的解法是：**在 Skill 文档中明确声明"Non-Negotiables"（不可妥协原则）**，包括"学习阶段不修改代码"。

### 难点 3：学习效果很难验证

传统的"让 Agent 讲给你听"方式，学完就忘。

这个项目的解法是：**用教练 Skill 做反向教学**，让 Agent 出题考你。

---

## 07 怎么用？手把手带你走完 10 轮学习

项目附带了一个完整的示例教程，用 **Kstheme/Study-Planner** 这个真实的开源项目做案例。

![](/images/repo-system-design/study-planner.png)

> Study-Planner 是一个 AI 学习计划生成工具，包含 Streamlit UI、Application Use Case、Domain Model、Agent Workflow、RAG、持久化，以及 Real/Fake 基础设施适配——本身就是非常适合练手的中等复杂度项目。

教程一共 **10 轮**，每一轮使用不同的 Skill，解决一个层次的问题。下面带你走一遍。

---

### 第 1 轮：仓库总览（repo-system-design-lab）

**你要输入的 Prompt：**

```
使用 $repo-system-design-lab 带我逐步学习 Kstheme/Study-Planner 的 system design。
这一轮只读分析，不要优化，也不要改代码。
请验证 README 和 docs/system-design.md 是否与源码一致，并输出：
1. 项目一句话定位
2. 主要模块和职责
3. 运行入口
4. 外部依赖
5. 最值得深入的 5 条核心链路
所有结论都必须引用源码文件。
```

**你得到的输出：**

Agent 会从 README 开始读，然后扫描目录结构，找到入口文件 `study_planner/interfaces/streamlit/app.py`，逐层了解：

- **应用层**（`study_planner/application/`）—— 用例编排
- **领域层**（`study_planner/domain/models.py`）—— 数据模型
- **Agent 层**（`study_planner/agents/planner_workflow.py`）—— Agent 工作流
- **基础设施层**（`study_planner/infrastructure/`）—— LLM、RAG、数据库适配器

输出中会明确标注哪些结论来自源码（Evidence）、哪些是推理（Inference）、哪些还不确定（Open Question）。

**✅ 这一轮你学到的是：** 从宏观上理解项目在做什么、用了什么技术栈、去哪里看核心代码。

---

### 第 2 轮：架构地图（repo-architecture-mapper）

**你要输入的 Prompt：**

```
使用 $repo-architecture-mapper 分析 Kstheme/Study-Planner。
请基于源码输出：
1. 分层架构图
2. 运行时 / 部署图
3. 数据所有权图
4. Fake 模式和 Real 模式切换图
并用源码证据解释每个模块边界。
```

**你得到的输出：**

Agent 会扫描整个项目，然后输出多个架构图（文字描述形式）：

**分层架构图：**

```
┌──────────────────────────────────┐
│  interfaces/streamlit/app.py     │  ← 页面渲染和用户交互
├──────────────────────────────────┤
│  application/                    │  ← 用例编排（GeneratePlanUseCase 等）
├──────────────────────────────────┤
│  domain/models.py                │  ← 数据模型，尽量无外部依赖
├──────────────────────────────────┤
│  agents/planner_workflow.py      │  ← Agent 工作流（profile→resource→planner→critic）
├──────────────────────────────────┤
│  infrastructure/                 │  ← LLM、RAG、DB、Fake/Real 切换
│   ├─ llm/   ├─ rag/   ├─ db/    │
│   └─ settings/                   │
└──────────────────────────────────┘
```

Agent 还会标注每个分层的源码位置，比如：

- `infrastructure/db/postgres/repositories.py` 里的 PostgresStudyPlanRepository 实现了 domain 定义的接口
- `infrastructure/llm/fake.py` 里的 FakeLLM 让你可以在不调用真实 API 的情况下测试

**✅ 这一轮你学到的是：** 项目的骨架和模块边界，知道每一层代码放哪里、为什么这样放。

---

### 第 3 轮：学习计划生成链路（repo-flow-tracer）

**你要输入的 Prompt：**

```
使用 $repo-flow-tracer trace "配置学习目标 -> 生成学习计划 -> 保存计划"。
请从 Streamlit 页面开始，一直追到 GenerateStudyPlanUseCase、PlannerWorkflow、
Domain Model、校验逻辑和持久化。
输出调用链表、sequence diagram、可靠性说明和有源码依据的设计取舍。
```

**你得到的输出：**

Agent 会从 `app.py` 里的 Streamlit 按钮事件开始，一步步追踪：

```
Streamlit 页面 (app.py:85) — 用户点击"生成计划"
  → GenerateStudyPlanUseCase (application/use_cases.py:42)
    → PlannerWorkflow.run() (agents/planner_workflow.py:30)
      → Profile 节点收集学习目标
      → Planner 节点生成计划草稿
      → Critic 节点校验并修复
      → Output 节点格式化结果
    → 校验 StudyPlan domain model 完整性 (domain/models.py:120)
  → StorageService 保存 (infrastructure/storage.py:55)
    → PostgresStudyPlanRepository 持久化
```

**每一步都带文件和行号引用。**

你不仅能读懂流程，还能看到设计取舍——比如 PlannerWorkflow 内部有个 **repair loop**：如果 Critic 节点发现计划有问题，会自动触发重试，最多 3 次。这是典型的 Agent 可靠性设计。

**✅ 这一轮你学到的是：** 一条完整功能链路从 UI 到数据库的全貌，以及沿途的工程决策。

---

### 第 4 轮：Agent Workflow 深度分析（repo-flow-tracer）

这是对上一轮的补充——不再看链路宽度，而是深入一个模块。

**你要输入的 Prompt：**

```
使用 $repo-flow-tracer 深入分析 PlannerWorkflow。
重点解释 profile、knowledge、resource、planner、critic、output 这些节点如何协作。
请说明 repair loop、Fake/Real LLM 切换、错误处理和 debug trace 的设计。
最后问我 5 个问题检查理解。
```

**你得到的输出：**

Agent 会深入 `agents/planner_workflow.py`，逐一拆解每个节点的输入输出：

| 节点      | 职责                           | 源码位置              |
| --------- | ------------------------------ | --------------------- |
| Profile   | 收集学习目标、时间、水平       | `agents/nodes.py:20`  |
| Knowledge | 检索相关领域知识               | `agents/nodes.py:50`  |
| Resource  | 推荐学习资源                   | `agents/nodes.py:80`  |
| Planner   | 生成学习计划草稿               | `agents/nodes.py:110` |
| Critic    | 校验计划质量，触发 repair loop | `agents/nodes.py:140` |
| Output    | 格式化最终输出                 | `agents/nodes.py:170` |

你还会学到：

- **Repair loop 如何工作**：Critic 发现计划缺少前置知识标注 → 通知 Planner 修正 → 重新校验，循环最多 3 次。如果第 3 次仍不合格，返回最后一次结果并标记 warning。
- **Fake/Real 切换**：通过配置开关，不调用真实 LLM 也能测试整个 Workflow。
- **错误处理**：每个节点都有超时控制，单个节点失败不会让整个 Workflow 崩溃。

最后 Agent 会问你 5 个问题，检查你是不是真的理解了。

**✅ 这一轮你学到的是：** 一个生产级别的 Agent Workflow 工程实现，包括容错、重试和可观测性设计。

---

### 第 5 轮：RAG 链路（repo-flow-tracer）

**你要输入的 Prompt：**

```
使用 $repo-flow-tracer trace "资料上传 -> 解析 -> chunk -> embedding/index -> 检索 -> LLM 回答 -> citation"。
请解释 local services 和 real services 的边界，
以及 Milvus、PostgreSQL、LLM 分别承担什么角色。
```

**你得到的输出：**

Agent 会带你走完完整的 RAG 链路：

```
用户上传 PDF/文档
  → 文件解析 (infrastructure/parsers/)
    → 文本分块 Chunking (infrastructure/rag/chunking.py:30)
      → Embedding 向量化 (infrastructure/llm/embeddings.py:45)
        → 向量写入 Milvus (infrastructure/rag/vector_store.py:60)
          → 用户提问后向量检索 (infrastructure/rag/retriever.py:80)
            → LLM 生成回答 + 引用标注 (agents/rag_workflow.py:100)
```

**三种存储各司其职：**

- **Milvus**：向量数据库，存储文档 Embedding，做相似度检索
- **PostgreSQL**：关系数据库，存储用户信息、学习计划、任务状态
- **LLM**（本地或云端）：负责生成回答和引用

你还能看到 **Fake 模式下的替换方案**——本地测试时用简单的向量相似度替代 Milvus，用 MockLLM 替代真实模型调用。

**✅ 这一轮你学到的是：** RAG 系统的完整数据流和组件职责拆分。

---

### 第 6 轮：持久化和状态设计（repo-flow-tracer）

**你要输入的 Prompt：**

```
使用 $repo-flow-tracer 分析 Study-Planner 的状态和持久化设计。
重点看 Streamlit session_state、StorageService、InMemory repositories、Postgres repositories。
请回答：谁是 source of truth？哪些情况下状态可能不一致？应该如何优化？
```

**你得到的输出：**

这是最有"系统设计"味的一轮。Agent 会分析：

- **Streamlit session_state**：前端临时状态，页面刷新即丢失
- **InMemory repositories**：测试用，重启后数据消失
- **Postgres repositories**：生产环境的 source of truth

Agent 会指出一个 **值得注意的设计问题**：session_state 中的数据与数据库之间存在短暂不一致窗口——用户看到页面上的计划是 session_state 缓存的，但数据库可能已经被其他操作更新了。

**这是典型的系统设计面试话题——缓存一致性问题。**

Agent 还会给出优化方向：引入版本号机制或 WebSocket 实时同步。

**✅ 这一轮你学到的是：** 真实项目中的状态管理和一致性权衡——这是面试必考的内容。

---

### 第 7 轮：复盘与重排（repo-flow-tracer）

```
使用 $repo-flow-tracer trace "任务完成情况 -> 生成复盘 -> 判断是否需要重排 -> 生成调整计划 -> 保存"。
重点解释如何保护已完成任务、如何校验调整后的计划、哪些失败场景需要处理。
```

Agent 会分析 Study-Planner 的计划更新机制，重点解释：

- **已完成任务保护**：重排时已标记为完成的任务不会被覆盖或删除
- **计划调整校验**：新计划需要经过 Critic 验证，确保不遗漏前置知识
- **失败场景处理**：部分任务失败时，标记为"受阻"而不是"完成"，保留上下文

**✅ 这一轮你学到的是：** 增量更新的设计思路——不是每次修改都全量重算。

---

### 第 8 轮：系统设计评审（repo-design-optimizer）

**你要输入的 Prompt：**

```
使用 $repo-design-optimizer 对 Study-Planner 做系统设计评审。
先不要改代码。
请从性能、可靠性、扩展性、可维护性、可观测性、安全性 6 个维度找问题。
每个问题都要包含源码证据、影响、优先级、优化方案和验证方法。
```

**你得到的输出：**

Agent 会从 6 个维度逐个扫描，输出一个完整的评审表（每个问题都带源码行号）：

| 维度     | 问题                       | 源码证据                             | 影响                 | 优先级 |
| -------- | -------------------------- | ------------------------------------ | -------------------- | ------ |
| 性能     | RAG 检索无缓存             | `infrastructure/rag/retriever.py:45` | 相同问题重复调用 LLM | 高     |
| 可靠性   | Repair loop 无最大轮次限制 | `agents/planner_workflow.py:85`      | 极端情况可能死循环   | 高     |
| 可观测性 | 缺少结构化日志             | `infrastructure/llm/real.py:22`      | 线上问题排查困难     | 中     |
| 安全性   | API Key 硬编码风险         | `infrastructure/settings.py:10`      | 密钥泄露             | 高     |

每个问题都有具体的验证方案——比如"为 RAG 添加 LRU Cache，通过命中率 benchmark 验证"。

**✅ 这一轮你学到的是：** 怎么系统性地评审一个项目的架构质量——不只是找 Bug，而是从工程维度全面扫描。

---

### 第 9 轮：安全优化实验（repo-design-optimizer）

**你要输入的 Prompt：**

```
使用 $repo-design-optimizer 为排名最高的 1 个优化点设计安全实验。
请先检查 git 状态。
然后建议 branch 或 worktree 方案，并输出涉及文件、最小有效改动、测试命令、
benchmark/verification 和回滚方式。
先不要实现，等我确认。
```

Agent 会：

1. 先检查你的 git 状态，确保工作区干净
2. 设计一个安全实验方案——比如用新 branch 修复 Repair loop 的无限重试问题
3. 列出涉及的文件、最小改动量、测试方法、验证指标和回滚步骤
4. **等你确认后才开始改代码**

这是最重要的习惯：**先有验证计划，再动手改代码。**

**✅ 这一轮你学到的是：** 安全地进行架构优化实验的方法论——不是莽改，而是有计划、有验证、有回滚。

---

### 第 10 轮：反向教学（repo-design-coach）

**你要输入的 Prompt：**

```
使用 $repo-design-coach 考我 Study-Planner 的 system design。
一次只问一个问题。
我回答后，你用源码证据纠正我。
从 Level 1 开始，逐步到 Level 5。
```

**你得到的体验：**

Agent 会从最简单的 Level 1 开始问：

> **Level 1 问题：** Study-Planner 的运行时入口在哪里？它是什么类型的应用？

你回答后，Agent 会对照源码纠正你的理解偏差：

> **你的回答：** 入口在 app.py，是一个 Web 应用。
>
> **Agent 纠正：** 更准确地说，入口在 `study_planner/interfaces/streamlit/app.py:85`，是一个 **Streamlit 单页应用**（SSP），不是传统的前后端分离 Web 应用。Streamlit 的特点是页面状态由 `session_state` 管理，不需要手动写前端路由。这是选择 Streamlit 作为 Prototyping 工具的关键设计决策。

问题会逐步深入，一直到 Level 5：

> **Level 5 问题：** 如果 Study-Planner 要支持 1000 个并发用户，你觉得当前架构中最大的瓶颈是什么？你至少需要引用 3 个源码位置来支持你的判断。

**✅ 这一轮你学到的是：** 通过被考来发现自己的知识盲区——这是最有效的学习方式之一。

---

### 3 步开启你的学习之旅

以上 10 轮不需要一天做完。**建议一周 2-3 轮，每轮 30-45 分钟。**

**第 1 步：安装 Skills 到目标项目**

```bash
cd your-project
mkdir -p .claude
cp -R path/to/repo-system-design-skills/claude/skills .claude/
```

**第 2 步：在 Claude Code 中输入**

```
使用 $repo-system-design-lab 学习这个项目的 system design。
先只读分析，输出架构地图，推荐 3 条核心链路。
所有结论必须引用源码文件。
```

**第 3 步：开始第一轮学习**

Agent 会从 README 开始，逐步扫描配置文件、目录结构、代码入口，然后输出有源码依据的架构分析。

---

## 08 总结

很多人在学系统设计时，都会掉进同一个坑：

**你读了书、背了题、看了别人的架构图，但你没有系统性地分析过自己写的代码。**

Repo System Design Skills 的价值不在于它有多智能，而在于它建立了一套可以重复使用的、有方法论支撑的**学习框架**。

- 它让 Agent 从"帮你写代码"变成"教你理解代码"；
- 它让学习从"听了就忘"变成"有据可查、有练有考"；
- 它让系统设计能力从"面试突击"变成"日常积累"。

**下次接手一个新项目，别急着写代码。**

先用这套 Skill 花 30 分钟理解它的架构。你学到的不仅是这个项目，而是理解任何项目的能力。

---

## 互动问题

> 你在学习系统设计时，最大的困难是看不懂源码、找不到学习主线，还是面试不知道怎么说？

---

## 参考资料

- Repo System Design Skills（GitHub）：项目地址，详见当前仓库
- Kstheme/Study-Planner：https://github.com/Kstheme/Study-Planner
