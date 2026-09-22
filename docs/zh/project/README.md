---
title: 项目
---

一个我构建和维护的开源项目集合。

<VPCardGrid :cols="{ sm: 1, md: 2, lg: 2 }">

<VPCard title="Study-Planner" icon="material-symbols:calendar-clock-outline">

AI 驱动的学习操作系统，把目标转化为结构化的学习计划。基于 LangGraph 的六阶段 Agent 工作流（Profile → Knowledge → Resource → Planner → Critic → Output），通过 Planner-Critic 修复循环持续优化计划的结构完整性与可执行性；完整 RAG 管线支持 PDF/TXT/Markdown 解析、Milvus 向量检索与引用溯源，可做材料问答、摘要生成、知识点提取与闪卡生成；另有 PostgreSQL 持久化层与任务分布、日负载、复习进度等分析看板。采用六层架构，领域模型零框架依赖，Fake/Real 双模式一行配置切换。**30+ pytest 用例。**

<Badge type="tip" text="Python" />
<Badge type="tip" text="LangGraph" />
<Badge type="tip" text="RAG" />
<Badge type="tip" text="Milvus" />
<Badge type="info" text="⭐ 6" />

**[GitHub →](https://github.com/Kstheme/Study-Planner)**

</VPCard>

<VPCard title="repo-system-design-skills" icon="material-symbols:code-blocks-outline">

从真实代码库学习系统设计的可复用技能模块。映射架构、追踪数据流、指导设计决策——同时支持 Claude Code 和 Codex。

<Badge type="tip" text="Markdown" />

**[GitHub →](https://github.com/Kstheme/repo-system-design-skills)**

</VPCard>

<VPCard title="nature-skills" icon="material-symbols:auto-stories-outline">

面向科研的可复用 AI 代理技能集合，专攻 Nature 风格的学术写作与科研绘图。我贡献了 **Nature-Paper-Card** 技能——生成有来源约束的深度 Paper Card（01–16 节），涵盖方法逻辑、证据链、结论边界和批判性分析，帮助科研人员深入理解论文并发现研究思路。

<Badge type="tip" text="Markdown" />
<Badge type="info" text="Fork" />

**[GitHub →](https://github.com/Yuan1z0825/nature-skills)**

</VPCard>

<VPCard title="Kstheme.github.io" icon="material-symbols:globe-outline">

基于 VuePress & vuepress-theme-plume 的个人博客与作品集，持续发布 LLM、Agent、RAG、AI4Math、计算机视觉与 AI 工程实践相关技术文章。同步运营微信公众号 **Kstheme for AI**（关注用户 250+）。

<Badge type="tip" text="TypeScript" />

**[GitHub →](https://github.com/Kstheme/Kstheme.github.io)**

</VPCard>

</VPCardGrid>
