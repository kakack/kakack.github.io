---
layout: post
tags: [LLM, Agent, Engineering]
title: Do Your Own Harness Engineering
date: 2026-05-12
author: Kyrie Chen
comments: true
toc: true
pinned: false
---

## Intro

`Harness Engineering`这个词最早是 `HashiCorp` 联合创始人 `Mitchell Hashimoto` 在 2026 年 2 月的博客中最早提出的。他的原话是：Agents 如果第一次就能给出正确的结果，或者至少只需稍作修改，效率会更高。实现这一目标最可靠的方法是为 agents 提供快速、高质量的工具，以便在出错时自动发出警报。对此，他总结为两种形式：

- 1.  **Better implicit prompting (AGENTS.md)**：对于一些简单的问题，例如代理重复运行错误的命令或找到错误的 API，请更新 AGENTS.md （或等效文件）。
- 2. **Actual, programmed tools.**： 例如，用于截屏、运行筛选测试等的脚本。通常需要修改 AGENTS.md 文件，以告知系统这些工具的存在。

每次 agents 犯错，用户都会尽力阻止它们再次犯同样的错误，或者反过来，也会尽力确保 agents 能够证明它们正在做对的事情。

## What is Harness Engineering

**Harness engineering（束具 / 护栏工程）**指的是：围绕 LLM agent 刻意设计并持续演进的一整套**运行时外壳与工程实践**——把「模型 + 提示」之外、却决定能否在真实环境里**可靠、可验证、可维护**地跑长任务的那一层补齐。它通常包括：工具与协议（读写仓库、执行命令、调用 API）、上下文与记忆策略、权限与审批、guardrails、可观测性与 trace、失败后的回放与修复回路，以及把团队规范写进仓库的隐式约束（如 `AGENTS.md`）和与之配套的自动化检查。

和单一框架或单次 prompt 不同，harness 强调：**把错误当成信号**——通过文档、脚本、测试与策略，让同类错误难以再次发生，或让 agent 能自动发现自己做错了。业界也有人把 agent harness 类比为「操作系统」：模型近似 CPU，上下文窗口近似 RAM，而 **harness 负责调度、驱动与资源管理**，agent 则是在其上运行的「应用」逻辑。

## Why Harness Engineering

`Anthropic`做过一些验证实验总结通用 agents 使用过程中的失败场景：

1. **追求一步到位（one-shotting）**：给定高层目标（例如「仿一个完整产品」）时，agent 倾向一次性铺开过多实现；容易在单次上下文窗口用尽前半途而废，下一会话面对的是**半成品且无清晰交接说明**的仓库，需要先「考古」才能把基础链路重新跑通（即便有 compaction，也未必总能把边界交代清楚）。
2. **过早宣布胜利（early-victory）**：项目中后期，新会话只看到「已经写了不少」，就**误判整个规格已交付**，在用户真正要求的完成度与质量远未达标时收口停手。
3. **过早标记功能完成**：在未做充分校验（尤其缺少**端到端、贴近真实用户路径**的验证）时就把条目标成通过；可能跑过单元测试或局部 `curl`，却漏掉只有通过浏览器/全流程才能暴露的问题。
4. **环境启动困难**：每开一个**无记忆**的新会话都要重新弄明白「如何在本地起服务、依赖怎么装、从哪里测」——既耗 token，也拉高把应用留在**不可用或难复现状态**的概率；若没有可重复的 `init` 与健康检查起手式，这种问题会反复出现。（以上归纳见 Anthropic 《[Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)》文中的实验观察与 Problem/Solution 表。）

### 失败的真实成本

这些失败场景不仅影响开发效率，更会带来实际的、可量化的成本：

- **时间成本**：每次环境重启失败或上下文丢失，平均需要 15-30 分钟的"考古"工作来重建状态。对于复杂项目，这个时间可能翻倍。当 agent 陷入 one-shotting 陷阱时，用户往往需要花费数小时清理半成品代码，甚至不得不回滚重来。
- **Token 成本**：重复的环境诊断、错误重现、上下文重建会消耗 2-3 倍的 token。以 Claude Opus 为例，一次失败的长任务可能消耗数百万 token，而其中大部分是在"空转"——反复尝试错误的命令、重新理解已经做过的工作。
- **信任成本**：过早宣布胜利和缺少验证会导致用户对 agent 的输出产生怀疑，不得不对每个结果进行人工复查。这种"信任赤字"会让协作效率大幅下降，最终用户可能退回到"只让 agent 做简单任务"的保守模式。
- **技术债务**：半成品代码、缺少测试的功能、未文档化的配置会在代码库中累积。这些问题在短期内可能不明显，但会在后续迭代中引发连锁反应——新功能难以集成、重构风险增大、onboarding 新成员的成本上升。

更隐蔽的成本在于**机会成本**：当团队把时间花在修复 agent 的错误上时，就无法投入到真正有价值的创新工作中。Harness Engineering 的价值不仅在于"减少错误"，更在于"释放人的时间去做 agent 做不了的事"。

### 根本原因：Agent 的认知盲区

这些失败场景背后有三个根本原因，它们共同构成了 agent 的"认知盲区"：

**1. 缺少外部记忆（No Persistent Memory）**

Agent 的上下文窗口虽然在不断扩大（从 4K 到 200K+），但本质上仍然是"短期记忆"。一旦会话结束或上下文被压缩，关键信息就会丢失：
- 上一次为什么选择这个技术栈？
- 哪些方案已经尝试过但失败了？
- 当前项目的架构约束和团队规范是什么？

人类开发者通过文档、注释、commit message 来构建"外部记忆"，但 agent 往往缺少这种机制。它每次启动都像是"失忆患者"，需要重新推理环境状态。

**2. 缺少验证能力（No Self-Verification）**

Agent 无法可靠地判断"做对了"还是"做错了"。它可能：
- 看到代码编译通过就认为功能完成（忽略运行时错误）
- 看到单元测试通过就宣布胜利（忽略集成问题）
- 看到命令执行成功就继续下一步（忽略输出中的警告信息）

这不是模型能力的问题，而是**缺少结构化的验证协议**。人类开发者会主动运行测试、检查日志、在浏览器中点击功能，但 agent 需要被明确告知"如何验证"。

**3. 缺少环境感知（No Environment Awareness）**

Agent 对运行环境的状态缺乏可靠的感知手段：
- 依赖是否已正确安装？版本是否匹配？
- 配置文件是否存在？环境变量是否设置？
- 服务是否在运行？端口是否被占用？

它只能通过执行命令并观察输出来"猜测"环境状态，这种间接感知容易出错。当环境配置复杂时（多个微服务、多个数据库、多个配置文件），agent 很容易陷入"试错循环"——反复尝试错误的命令，却无法系统性地诊断问题根源。

**Harness Engineering 的本质，就是为 agent 补齐这三种能力：**

- 通过 **AGENTS.md / CLAUDE.md** 提供外部记忆
- 通过 **验证脚本和测试工具** 提供自我验证能力
- 通过 **环境检测和健康检查** 提供环境感知能力

这不是"锦上添花"的优化，而是让 agent 从"能跑 demo"到"能上生产"的**必要基础设施**。

## How to Build Effective Harness

### 文档驱动的隐式约束（AGENTS.md / CLAUDE.md）



### 可编程工具与验证脚本



### 上下文管理与记忆策略



### 环境初始化与健康检查



### 可观测性与失败回放



## Best Practices & Patterns

### 针对 One-shotting 的防护策略



### 针对 Early-victory 的验证机制



### 针对环境问题的标准化流程



## Case Study



## Conclusion



---

# Reference

- [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Harness Engineering - first thoughts](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering-memo.html)
- [My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey)
