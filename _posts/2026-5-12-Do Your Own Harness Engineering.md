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

### Real Cost

这些失败场景不仅影响开发效率，更会带来实际的、可量化的成本：

- **时间成本**：每次环境重启失败或上下文丢失，平均需要 15-30 分钟的"考古"工作来重建状态。对于复杂项目，这个时间可能翻倍。当 agent 陷入 one-shotting 陷阱时，用户往往需要花费数小时清理半成品代码，甚至不得不回滚重来。
- **Token 成本**：重复的环境诊断、错误重现、上下文重建会消耗 2-3 倍的 token。以 Claude Opus 为例，一次失败的长任务可能消耗数百万 token，而其中大部分是在"空转"——反复尝试错误的命令、重新理解已经做过的工作。
- **信任成本**：过早宣布胜利和缺少验证会导致用户对 agent 的输出产生怀疑，不得不对每个结果进行人工复查。这种"信任赤字"会让协作效率大幅下降，最终用户可能退回到"只让 agent 做简单任务"的保守模式。
- **技术债务**：半成品代码、缺少测试的功能、未文档化的配置会在代码库中累积。这些问题在短期内可能不明显，但会在后续迭代中引发连锁反应——新功能难以集成、重构风险增大、onboarding 新成员的成本上升。

更隐蔽的成本在于**机会成本**：当团队把时间花在修复 agent 的错误上时，就无法投入到真正有价值的创新工作中。Harness Engineering 的价值不仅在于"减少错误"，更在于"释放人的时间去做 agent 做不了的事"。

### Agent's Blind Spots

这些失败场景背后有三个根本原因，它们共同构成了 agent 的"认知盲区"：

**1. 缺少外部记忆（No Persistent Memory）**

Agent 的上下文窗口虽然在不断扩大（从 4K 到 200K+），但本质上仍然是"短期记忆"。一旦会话结束或上下文被压缩，关键信息就会丢失：上一次为什么选择这个技术栈？哪些方案已经尝试过但失败了？当前项目的架构约束和团队规范是什么？

Dex Horthy 有过一个有趣的实验发现：**上下文填得越满，LLM 输出质量越差。** 以 168K token 的上下文窗口为例，大约用到 40% 就开始走下坡路了：前 40% 的 context window 可以视作一个 smart zone，聚焦准确精炼的信息；后 60% 是一个 dumb zone，充满了幻觉、循环和错误。

人类开发者通过文档、注释、commit message 来构建"外部记忆"，但 agent 往往缺少这种机制。

**2. 缺少验证能力（No Self-Verification）**

Agent 无法可靠地判断"做对了"还是"做错了"。它可能看到代码编译通过就认为功能完成（忽略运行时错误）、看到单元测试通过就宣布胜利（忽略集成问题）、看到命令执行成功就继续下一步（忽略输出中的警告信息）。

这不是模型能力的问题，而是**缺少结构化的验证协议**——人类开发者会主动运行测试、检查日志、在浏览器中点击功能，但 agent 需要被明确告知"如何验证"。

**3. 缺少环境感知（No Environment Awareness）**

Agent 对运行环境的状态缺乏可靠的感知手段：依赖是否已正确安装、配置文件是否存在、服务是否在运行、端口是否被占用。它只能通过执行命令并观察输出来"猜测"环境状态，这种间接感知容易出错，在多服务、多配置的复杂环境下尤其容易陷入"试错循环"。

Harness Engineering 要解决的，就是为 agent 补齐这三种能力——这不是"锦上添花"的优化，而是让 agent 从"能跑 demo"到"能上生产"的**必要基础设施**。下一节就来看具体怎么做。

## How to Build Effective Harness

Anthropic 在其工程博客 [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) 中提出了一种"两阶段 agent"的实践：先用一个 **initializer agent** 准备好工作环境与任务清单，再让 **coding agent** 在每个会话中按既定流程推进。下面的实践方法主要参考自这一文章中明确提到的做法，分别对应前一节列出的三种认知盲区。

### Document Implicit Constraints

`AGENTS.md`（或 `CLAUDE.md`、`.cursor/rules` 等等价文件）是 harness 中最基础也最重要的一层：它把"team knowledge"沉淀进仓库，让 agent 在每次会话开始时都能读到这些约束。

可以放进 `AGENTS.md` 的内容包括：

- **环境启动方式**：如何启动开发服务器、如何运行测试、依赖如何安装
- **项目结构与代码风格约定**：目录布局、命名规范、import 顺序
- **常见陷阱与禁区**：哪些命令不要跑、哪些目录不要动、哪些 API 已经废弃
- **任务进度的存放位置**：进度文件、特征列表文件在哪里

一个关键原则是：**每当 agent 反复犯同一个错误，就把规则补进 `AGENTS.md`，而不是反复在 prompt 里纠正。**

### Verification Script 

文档只能提供"提示"，真正能让 agent 自我验证的是**可执行的工具与脚本**。Anthropic 在博客中描述的做法是：

- **`init.sh`**：一个用于启动开发服务器的脚本。每次 coding agent 开始会话时都会先跑一遍 `init.sh`，确认基础链路可用，再开始新功能开发。
- **特征列表文件（feature list）**：用 JSON 格式记录数百条待实现的功能，每条都包含描述与**逐步的验证步骤**。文件被标记为"不允许 agent 删除或编辑测试"，避免 agent 通过删除断言来"伪造通过"。
- **浏览器自动化工具**：博客中提到使用 Puppeteer MCP 这类工具做端到端验证，让 agent 像真实用户一样在浏览器中操作，而不是只跑单元测试或 `curl`。

> 工具的选择不是关键，关键是：**让"是否完成"这件事，从 agent 的主观判断变成可执行的客观检查**。

### Context & Memory

把状态外置到文件、而不是堆在上下文里，本身就是缓解前面提到的"smart zone 退化"的直接手段。Anthropic 的做法是引入一份持续维护的进度文件，并结合 git：

- **`claude-progress.txt`**：由 initializer agent 创建，coding agent 在每次会话结束前把"做了什么、下一步要做什么、有哪些已知问题"写进去。下一个会话开始时，第一件事就是读取这份文件。
- **Git 作为外部记忆**：每完成一个 feature 就立即 commit，并写清晰的 commit message。当出现问题时，可以通过 `git revert` 快速回滚到已知良好状态。
- **结构化任务清单**：JSON 格式的特征列表初始时全部标记为 "failing"，agent 每完成并验证一条才能把它改为 "passing"。这强迫 agent 在每一步都有明确的目标与产出。

[Vasilopoulos et al. (2026)](https://arxiv.org/abs/2602.20478) 将上下文形式化为三层：`热记忆（Hot Memory）`、`领域专家（Domain Experts）`、`冷记忆知识库（Cold-Memory Knowledge）`。然后分成三个上下文体系层级：

| 层级 | 加载时机 | 内容示例 | 上下文占用 |
| --- | --- | --- | --- |
| Tier 1：会话常驻 | 每次会话自动加载 | `AGENTS.md` / `CLAUDE.md`、项目结构概览 | 最小 |
| Tier 2：按需加载 | 特定子 Agent 或技能被调用时 | 专业化 Agent 的上下文、领域知识 | 中等 |
| Tier 3：持久化知识库 | Agent 主动查询时 | 研究文档、规格说明、历史会话 | 按需 |



### Environmental Initialization & Health Check

针对 "环境启动困难" 这一失败场景，Anthropic 给出的对策是把会话起手式标准化：

- **固定的启动序列**：每个 coding agent 会话开始时跑 `pwd`、读取 git log 与 progress 文件、挑选优先级最高的未完成特征、运行 `init.sh` 确认环境可用——这一整套动作是写死的，不依赖 agent 即兴发挥。
- **健康检查先行**：在写新代码之前，先验证基础链路是否工作。如果 `init.sh` 都跑不起来，就先修环境，而不是叠加新功能。
- **可重复性**：`init.sh` 的存在意味着任何一个新会话（甚至新的开发者）都能用同一条命令把项目跑起来，避免"只在我机器上能跑"的窘境。

### Failure Monitor & Replay

要让 harness 真正可演进，需要看得见 agent 在做什么、哪里卡住：

- **进度文件作为审计轨迹**：`claude-progress.txt` 不仅是给下一个会话看的，也是给人类看的——能快速看到 agent 在哪些 feature 上反复失败，从而判断是 prompt 问题、工具问题还是任务本身定义不清。
- **Git 历史作为回放手段**：每个 feature 一个 commit 的好习惯，让"回到上一个能跑的状态"变成一行命令的事，也方便人类 review agent 的决策路径。
- **特征列表的状态转移**：通过观察哪些 feature 长期停留在 "failing"，可以发现 harness 本身的短板（例如某类任务缺少合适的验证工具）。

这一层的核心思想是：**harness 不是一次性写完的，而是从 agent 的失败中持续演进出来的**。可观测性决定了你能多快从失败里学到东西。

## Best Practices & Patterns

前一节按"组件"维度拆解了 harness 的构成，这一节换一个视角：**针对前面 Why 一节列出的失败场景，逐一给出对应的防护模式**。这些模式同样以 Anthropic 博客中 Problem / Solution 表格中明确记录的做法为基础。

### One-shotting 

One-shotting 的根源在于 agent 把"大目标"当作一次会话内可完成的任务，结果在上下文耗尽前留下一堆半成品。可落地的防护手段有：

- **用 initializer agent 先拆任务，再让 coding agent 干活**：Anthropic 的做法是把"理解需求 + 生成 feature list"和"实现 feature"分给两个不同角色。前者负责把高层目标拆成数百条 **粒度小、可独立验证** 的特征，后者每个会话只挑一条来做。
- **把特征列表写死成 JSON，禁止 agent 删改测试**：JSON 格式（而非 Markdown）能让 agent 不太轻易"自由发挥"地改写定义；同时配上"不允许删除或编辑测试"的明文规则，避免 agent 通过修改验收标准把任务"压缩进一个会话"。
- **每条特征都带逐步验证步骤**：这意味着 agent 不能在一个会话里"草草扫一遍所有特征"，因为每一条都要走完独立的验证流程才能改状态。
- **强制 commit 粒度**：每完成一条 feature 就 commit 一次，让"工作进度"沉淀到 git 历史而不是上下文窗口里。这样即使会话中断，下一次也能从 commit 边界继续，而不是落在一个"改了一半"的中间状态。

核心思路是：**让 agent 没有机会一步到位**——把大任务在 harness 层就切成小片，agent 每次只能啃一片。

### Early-victory

Early-victory（过早宣布胜利）和过早标记功能完成，本质上都是 agent "自己批改自己的作业"。对策是把"判定是否完成"这件事从 agent 主观判断挪到客观可执行的检查上：

- **特征列表初始全部为 "failing"**：Anthropic 明确提到这一点——所有 feature 的初始状态都是失败，agent 必须走完该 feature 配套的验证步骤，才能把状态改成 "passing"。"还没验证"和"已通过"在状态上是不同的，避免被混为一谈。
- **验证步骤必须是端到端的真实路径**：博客中强调使用 **Puppeteer MCP** 这类浏览器自动化工具，像真实用户一样点按钮、填表单、看页面跳转，而不是只跑单元测试或 `curl`。前面 Why 一节提到的"过早标记功能完成"问题，绝大多数都发生在"单测过了但浏览器里点不通"的场景。
- **禁止 agent 自行修改验证逻辑**：如果 agent 可以在跑不通时改测试来让它通过，验证机制就形同虚设。把"不允许删除或编辑测试"作为硬规则写进 AGENTS.md 与特征列表文件。
- **`init.sh` 作为前置健康检查**：每个会话开始前都要先跑通 `init.sh`，确认基础链路 OK。如果连 `init.sh` 都跑不起来，所谓"已完成的特征"很可能根本没在工作的环境里被验证过。

一句话总结：**让"通过"成为一个外部系统能认定的事实，而不是 agent 写在 progress 文件里的一行字**。

### Environment Issue

"环境启动困难"在长任务里之所以代价高，是因为每个无记忆的新会话都要把"怎么把项目跑起来"重新摸索一遍。Anthropic 给出的对策是把启动流程标准化、写死：

- **固定的会话起手序列**：每个 coding agent 会话开始时都执行同一组动作——`pwd` 确认所在目录、读取 git log 与 `claude-progress.txt`、挑选优先级最高的未完成特征、运行 `init.sh`。整个序列由 harness 强制执行，不依赖 agent 即兴发挥。
- **`init.sh` 作为单一入口**：把"启动开发服务器"这件事收敛到一个脚本里，确保任何会话（任何开发者）都能用同一条命令把项目跑起来，消灭"只在我这能跑"的环境分歧。
- **把环境约束写进 AGENTS.md**：依赖版本、必需的环境变量、常见的踩坑命令、哪些目录不能动——这些规则写在仓库里，agent 每次会话都能读到，不需要在 prompt 里反复交代。
- **先修环境再写代码**：如果 `init.sh` 跑不起来，会话的第一优先级是修环境而不是加新功能。把这条规则明文写进 AGENTS.md，可以避免 agent 在"环境其实是坏的"前提下继续叠功能。

这一组做法的共同特征是：**把"如何把项目跑起来"从 agent 每次都要重新推理的开放问题，变成 harness 已经替它答好的封闭问题**。这也直接缓解了前面提到的"前 40% smart zone"被环境诊断浪费掉的窘境——把宝贵的上下文留给真正的开发工作。

## Case Study

下面挑两个公开报道过的实践来对照前面讲的思路。它们的工程细节差异很大，但放在 harness 的视角下能看到相同的核心模式。

### Anthropic：Initializer + Coding Agent 的两阶段流水线

Anthropic 在 [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) 中描述的实验，把一个原本会一步到位失败的长任务拆成了"两个 agent 角色 + 一组固定文件"：

- **Initializer agent**：负责理解整体需求，生成 `init.sh`、`claude-progress.txt`、特征列表 JSON 文件，并做一次初始 commit。它的输出是给后续会话用的"启动包"，不直接写功能代码。
- **Coding agent**：每个会话都跑同一套起手式——`pwd` → 读 git log → 读 progress 文件 → 挑一条 "failing" 状态的特征 → 跑 `init.sh` 验证环境 → 开始实现 → 走完该特征的验证步骤 → commit → 更新 progress 文件。
- **强制规则**：特征列表里"不允许 agent 删除或编辑测试"，避免 agent 通过修改验收标准来"伪造通过"。

对照前面 Why 一节的四种失败场景：one-shotting 被"两阶段 + 单次会话只挑一条特征"切碎了；early-victory 被"特征初始为 failing、需配套验证才能改 passing"挡住了；环境启动困难被"`init.sh` + 固定起手序列"消解了。这是一个把 harness 的几条原则做齐的最小可复用案例。

### OpenAI：Codex 在百万行内部项目里的 harness

Martin Fowler 的 [Harness Engineering - first thoughts](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering-memo.html) 转述了 OpenAI 内部用 Codex 构建一个项目的做法：5 个月、超过百万行代码、"没有手写代码"。其 harness 被归纳为三类组件：

- **上下文工程**：持续演进的代码库知识库，配合 agent 在需要时按需调取可观测性数据与浏览器导航能力。
- **架构约束**：一组**确定性的自定义 linter 和结构测试**，由 LLM agent 监控；这是把"是否符合架构规范"这件事从 agent 主观判断挪到了可执行检查上。
- **垃圾回收**：定期运行的 agent 专门去找文档不一致、架构约束违规，把"漂移"控制在低水位。

关键的工作流不是某个 agent 多强，而是**反馈回路**：当 agent 在某类任务上反复出错时，识别出"缺什么"——工具、护栏还是文档——把这部分补回到代码库里，下次同类问题就被 harness 自动接住了。

### Common Mode

抛开规模差异，两个案例在 harness 设计上呈现出几乎一致的骨架：

| 维度 | Anthropic 实验 | OpenAI Codex 项目 |
| --- | --- | --- |
| 外部记忆 | `claude-progress.txt` + git | 持续维护的代码库知识库 |
| 客观验证 | JSON 特征列表 + Puppeteer | 确定性 linter + 结构测试 |
| 环境/约束闭环 | `init.sh` + AGENTS.md 规则 | 自定义工具 + 架构约束 |
| 演进机制 | 同类错误补进 AGENTS.md | 垃圾回收 agent + 反馈到仓库 |

差异主要在"做到什么粒度"，而不是"做什么"。harness 工程化程度的高低，最终决定了 agent 能在多长的时间跨度和多大的代码体量上稳定工作。

## Conclusion

回到开头 Mitchell Hashimoto 那句话：让 agent 第一次就做对的最可靠方法，是给它高质量的工具与文档，并在出错时主动报警。这篇文章本质上就是把这个直觉展开来：

- **Why**：通用 agent 在长任务里会反复掉进 one-shotting、early-victory、过早标记完成、环境启动困难这四个坑，根源是它缺少外部记忆、自我验证和环境感知三种能力，并且为此付出可观的时间、token、信任与技术债成本。
- **How**：用 `AGENTS.md` 沉淀团队知识、用 `init.sh` 与特征列表把"是否能跑、是否完成"变成可执行检查、用 progress 文件和 git 做跨会话的外部记忆、用可观测性把 harness 自己的短板暴露出来。
- **Patterns**：每条失败场景都有对应的防护模式，关键是**把判断权从 agent 挪到 harness**——切碎大任务、客观化验证、固化环境流程。
- **Case**：从 Anthropic 的两阶段流水线到 OpenAI 的百万行项目，骨架是相同的：外部记忆、客观验证、闭环约束、持续演进。

值得强调的一点是：**harness 不是写一次就完事的基础设施，而是从 agent 的失败中持续长出来的有机体**。每一次 agent 走错路，都是一次让 harness 变得更好的信号——把那次错误的修复"产品化"（写进文档、写成脚本、加进 linter），同类错误就再也不必由人去盯。

当 agent 的能力还在快速演化时，harness 是少数能被自己掌握、能持续积累的杠杆。"Do your own harness engineering" 的真正含义是：**不要等通用 agent 变得完美，先把你自己的工作环境改造成 agent 能稳定工作的样子**。

---

# Reference

- [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Harness Engineering - first thoughts](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering-memo.html)
- [My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey)
