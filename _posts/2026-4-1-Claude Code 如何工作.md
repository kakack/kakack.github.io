---
layout: post
tags: [LLM, Agent, Engineering]
title: Claude Code 如何工作
date: 2026-04-01
author: Kyrie Chen
comments: true
toc: true
pinned: false
---

2026 年 3 月 31 日，Anthropic 发布的 `@anthropic-ai/claude-code` v2.1.88 里多了一个 59.8 MB 的 `.js.map` 文件。Bun 默认生成 source map，而 `.npmignore` 里漏掉了对应的忽略规则。几小时内，约 1,900 个 TypeScript 文件、512,000 行代码被全网镜像。

但真正的故事不是泄露本身，而是工程。

大多数报道都在追逐猎奇细节：隐藏功能、内部代号、卧底模式。这些噪音会随时间消散，六个月之后依然有价值的是架构。Claude Code 是目前世界上使用最广泛的 AI 编程 Agent，它的源码揭示了一套经过大规模生产验证的 hard-won patterns。本文将 Particula Tech 的架构解读与虎嗅的五层架构分析相融合，试图说清这件真正重要的事：**Claude Code 的引擎盖下，究竟是什么样子。**

## 五层架构，从外到内

很多人以为 AI 编程工具不过是给模型 API 套一层终端界面。但 512,000 行代码、约 40 个内置工具、85 个斜杠命令、React+Ink 渲染的终端 UI、Bun 运行时——这分明是一个完整的生产级系统，不是套壳。

从源码中可以看出，整个系统被清晰地划分为五层：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-1.png)

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-8.png)
*Claude Code 的系统架构｜reddit*

| 层级 | 核心职责 |
| --- | --- |
| **入口层** | CLI、桌面端、网页、IDE 插件、SDK 的统一路由 |
| **运行层** | REPL 循环、状态机、Hook 系统、任务队列 |
| **引擎层** | QueryEngine、动态提示词组装、流式响应、上下文压缩 |
| **工具与能力层** | 约 40 个内置工具、权限隔离、MCP 扩展、安全校验 |
| **基础设施层** | 认证、缓存、文件存储、远程控制、telemetry、多代理隔离 |

这五层的关系不是简单的上下堆叠，而是围绕一个核心信条组织的：**把 Agent loop 做得极简，把 loop 周围的 harness 做得极重。** 512,000 行代码里，Agent 循环本身可能只有 20 行；剩下的全是上下文管理、权限系统、工具协议、错误恢复、记忆与压缩。

## 入口层（Entrypoints）

入口层的核心任务是：把用户侧碎片化的输入标准化，再向下传递。`entrypoints/cli.tsx` 会在极早期做 fast-path 分流：`--version`、MCP server 辅助模式、bridge / remote control、daemon、headless runner 等特殊模式会在这里短路，避免加载完整的 CLI。

`main.tsx`（约 4,683 行）则是真正的运行时编排器。它负责解析 CLI 参数、加载配置与策略、初始化 GrowthBook feature flags、处理 migrations，并选择 interactive / non-interactive / remote / direct-connect / assistant 模式。这说明 Claude Code 从一开始就被设计为多端产品，前端的变化不会污染核心逻辑。

## 运行层（Runtime）

运行层管理着每条命令的进出和状态更新。它的核心是一个极简的 TAOR 循环：Think -> Act -> Observe -> Repeat。如果把它翻译成代码，本质上就是这样一个 `while` 循环：

```typescript
while (true) {
  const response = await model.generate(messages);
  if (!response.hasToolCall()) break;
  const result = await executeTool(response.toolCall);
  messages.push(response, result);
}
```

模型产生一条消息。如果包含工具调用，就执行工具，把结果追加到对话历史中，然后继续循环。没有工具调用？循环停止，等待用户输入。所有状态都以不可变的 message 形式存在——仅从对话历史就能完整重建状态。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-2.png)

这个循环听起来简单到令人惊讶。但这就是重点：很多团队在构建第一个 Agent 时会去搞复杂的状态机、DAG 编排器或自定义控制流引擎。Claude Code 证明你不需要这些。一个基于 tool call 的 `while` 循环，加上 message history 作为核心数据结构，就能覆盖绝大多数 Agentic 行为。

复杂度不在循环里，而在循环周围的 harness 中。运行层的 REPL（约 5,005 行）不是"把文本打印出来"那么简单，它还要处理流式输出、权限对话框、tool progress、任务前后台切换、远程会话状态、消息滚动与恢复。可以把它理解成**终端里的会话操作系统**。

## 引擎层（Engine）

引擎层是系统的心脏，核心是一个 QueryEngine 单例，负责拼接上下文、管理提示缓存、处理流式响应和压缩对话，代码量接近 46,000 行。

### 动态提示词组装

Claude Code 没有一份静态的 system prompt。相反，它是**数百个提示碎片在运行时动态拼装**的结果。根据模式、工具、上下文的不同，系统会注入不同的提示片段。光是安全守则就有约 5,677 个 token，相当于两万字的行为规范。这是把软件工程的模块化思想搬进了提示词管理。

### SYSTEM_PROMPT_DYNAMIC_BOUNDARY

引擎层使用了一个 `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` 模式，把静态指令（会话间不变）与动态上下文（会话相关）显式分离。这不仅是组织上的优化，更是**缓存优化**：静态内容命中 prompt cache，动态内容放在边界之后。团队甚至跟踪了 **14 个独立的 cache-break vectors**，监控到底是什么在让缓存失效。

### 三层上下文压缩

引擎层对上下文管理的重视程度，从三层压缩架构就能看出来：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-5.png)

- **MicroCompact**：在本地裁剪旧的 tool 输出，**零 API 调用**
- **AutoCompact**：在接近上下文上限时触发，生成最多 20,000 token 的结构化摘要，并带有**连续 3 次失败即断路的熔断器**
- **Full Compact**：全量压缩后，重新注入最近访问的文件（每文件上限 5,000 token）、活跃计划和 skill schema

据说一个 autocompaction 无限重试的 bug，曾经每天烧掉约 **250,000 次 API 调用**——上下文管理决策在这个规模上直接等于真金白银。

### Plan Mode 与 Coordinator Mode

引擎层里还有两个多代理机制，但它们的定位截然不同：

**Plan Mode** 是只读分析模式。Agent 用只读操作探索代码库、规划方案、不做任何修改。它的价值是在真正执行之前给出一个可以审查的计划，防止 AI 直接动手动坏了再说。

**Coordinator Mode**（`CLAUDE_CODE_COORDINATOR_MODE=1`）才是真正的并行多代理机制。开启后，一个 Claude 实例充当协调者（Coordinator），通过 `AgentTool` 创建并管理多个 Worker Agent，在独立的 git worktree 里隔离执行。协调者不自己写代码，只负责分配任务和汇总输出。Agent 之间的通信没有魔法消息总线，就是结构化文本直接 pipe 进协调者的上下文窗口。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-3.png)

最精妙的是经济性设计：子 Agent fork 时，会创建与父上下文**字节级一致的副本**，从而**共享 KV cache**。这意味着子 Agent 只需要为自己独有的指令付费，而不是重复支付整个共享上下文的 token。并行由此变得几乎免费。

## 工具与能力层（Tools&Caps）

约 40 个内置工具，每一个都是独立的、权限隔离的能力单元。工具基类的定义就有约 29,000 行 TypeScript。

### 把安全嵌入使用点，而不是放进策略文档

大多数 Agent 框架把安全交给单独的配置层：权限文件、guardrails、内容过滤器。Claude Code 的做法不同——安全规则直接嵌入在工具描述里，就在模型每次调用时能看到的位置。

比如 Bash 工具的 description 里明确写着：

```
- NEVER run destructive git commands (push --force, reset --hard,
  checkout ., restore ., clean -f, branch -D) unless the user
  explicitly requests these actions
- CRITICAL: Always create NEW commits rather than amending
```

LLM 在行动附近看到约束时，远比在遥远的 system prompt 段落里更不容易"遗忘"。

### 专用工具优于通用 Shell

Claude Code 本可以只暴露一个 `bash` 工具就完事，但它为常见操作都提供了专用工具。Bash 工具的 description 甚至会明确警告：不要用本工具来运行 `find`、`grep`、`cat`、`head`、`tail`、`sed`、`awk` 或 `echo`。

| 通用 Shell 命令 | Claude Code 专用工具 |
| --- | --- |
| `grep`, `rg` | `Grep`（带类型参数，底层用 ripgrep） |
| `find`, `ls` | `Glob`（模式匹配，结果排序） |
| `cat`, `head`, `tail` | `Read`（带行号，支持图片/PDF） |
| `sed`, `awk` | `Edit`（精确字符串替换，要求先 Read） |
| `echo >`, `cat <<EOF` | `Write`（修改现有文件前要求先 Read） |

这样做的原因有三点：**可观测性**（结构化调用产生结构化日志）、**安全性**（每个工具有独立的权限和校验逻辑）、**模型性能**（专用工具的显式参数和丰富描述能帮模型更快选对动作）。

### 六层权限防线

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-10.png)
*Agent 的六层防线与 `useCanUseTool.tsx`*

每一个工具调用在真正执行前，要经过六层检查，全部实现在 `useCanUseTool.tsx` 这个文件里。

1. **白名单过滤**：项目和用户配置直接过滤掉不在允许范围内的操作。
2. **自动模式分类器**：判断这个操作在无人值守的情况下是否安全。
3. **协调者门控**：针对 Coordinator 编排层做授权验证。
4. **Swarm 工作者门控**：针对子代理执行做授权验证。
5. **Bash 安全分类器**：23 条具体规则，覆盖 Zsh 等号扩展、Unicode 零宽字符注入、IFS 空字节注入等攻击向量——这些规则的具体程度说明 Anthropic 在真实运行中遇到过这些问题。
6. **交互式用户确认**：弹窗或桥接审批。

设计哲学是**每一层独立失败**。不是有了最终确认才放行，而是任何一层发现问题就停下来。纵深防御，不是单点守门。

## 基础设施层（Infrastructure）

除了认证、文件存储、缓存这些常规内容，基础设施层有几个容易被忽略但至关重要的设计。

### 14 个缓存断点与粘性锁存器

提示缓存架构用粘性锁存器（sticky latch）管理 14 个缓存断点，防止模式切换导致缓存失效。之所以要如此精细，是因为每一次缓存失效都在花真实的钱。

### 远程控制机制

GrowthBook 有远程 kill switch，可以针对特定用户禁用功能。Policy Limits 每小时轮询一次，服务端可以远程禁用工具或限制功能。企业版甚至支持远程推送 `settings.json` 覆盖本地配置。这意味着 Claude Code 的行为边界不完全由本地配置决定，也由 Anthropic 服务端实时定义。

### 原生客户端认证（cch Attestation）

HTTP 请求头里有一个占位符 `cch=00000`，在请求真正发出前会被 Bun 底层的 Zig 代码替换成一个计算出来的哈希值。服务端验证这个哈希，确认请求来自未被篡改的官方二进制文件。这套机制在 JavaScript/TypeScript 层以下，无法通过修改前端代码绕过。

更关键的一点：即使设置了 `DISABLE_TELEMETRY=1`，这个认证 token 依然会随每个 API 请求发出，无法关闭。遥测可以关，但身份认证关不掉。

## 这套架构，借鉴了人类的大脑

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-9.png)
*Claude Code 解读（HARNESS 视角）｜vrungta.substack*

从技术谱系上看，Claude Code 的 TAOR 循环脱胎自 ReAct（Yao et al., 2022），但在此基础上加了 MCP 协议、git worktree 隔离和多层记忆系统。Memory 分层的思路来自认知科学，Claude Code 做的事是把这些分散的工程想法整合起来，做成 51 万行代码——而且 90% 是用它自己写的。

Claude Code 的记忆系统分三层，与认知科学里的记忆分类高度对应：

- **Semantic 层**：稳定知识，类似长期语义记忆。只写入高信号内容，矛盾信息自动剔除，用 RAG 检索，不全量加载。
- **Episodic 层**：过去对话序列，类似情景记忆。按时间索引，按需检索。
- **Working 层**：当前任务的动态上下文窗口，类似工作记忆。超出限制时用指针代替内容，保持轻量。

核心信条永远是：**不要把所有东西都塞进上下文窗口，存索引，按需拉取。**

### 怀疑论记忆（Skeptical Memory）

`MEMORY.md` 是一个始终加载的轻量索引文件，通常不超过 200 行，每行是一个指向详细 topic file 的指针。真正详细的笔记分散在 `debugging.md`、`patterns.md` 等文件中，按需获取，而不是默认加载。

最关键的设计选择是：**Agent 被明确指示把记忆当作 hint，在行动前必须对照实际代码库进行验证**。一条记忆说"函数 X 在文件 Y 中"，Agent 会先 `grep` 确认它仍然存在，而不是直接相信。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-4.png)

这种"怀疑论记忆"解决了生产环境中反复出现的问题：Agent 因为记忆陈旧而自信地给出过时建议。修复方案不是更好的存储，而是把验证写进检索流程。

### Reflection 机制

每轮 Act 完成后、下一轮 Think 开始前，系统会插入一个自省环节：检查刚才的操作是否达到预期、是否陷入循环、是否遗漏约束。社区测试表明，加入 reflection 后任务成功率可以从 60% 提升到 85%。代价是多一轮模型调用的成本。

## 卧底模式、反蒸馏与原生认证

把泄露代码中的信息控制机制放在一起看，Anthropic 为三个方向各设了一道防线：

**对外，防身份暴露。** `undercover.ts` 约 90 行。当 Anthropic 内部员工（`USER_TYPE==='ant'`）在非内部仓库操作时自动激活。效果是剥离所有相关标识：提交信息里没有模型名，没有 "Claude Code" 字样，没有 Co-Authored-By 署名。代码注释写着：没有强制关闭的开关。系统提示里甚至有一句话："你在卧底行动中。不要暴露你的身份。"

**对竞争对手，防被学习。** 当环境变量 `ANTI_DISTILLATION_CC` 打开时，服务端会向系统提示里注入一批**假的工具定义**。任何在 API 层面录制 Claude Code 流量、拿来训练竞争模型的人，会把这些假工具一并学进去。另有辅助机制：把推理链替换成带加密签名的摘要，外部观测者看到的不是完整推理过程，而是一个无法逆向的摘要。

**对篡改客户端，防伪造服务。** 就是上面提到的 cch Attestation，Bun/Zig 层的认证机制，JS/TS 层无法模拟。

三件套的共同逻辑：**对外不露身份，对对手不露推理，对非官方客户端不提供服务。**

## Auto-Dream：代码里的REM睡眠

基础设施层里有一个后台进程叫 **Auto-Dream**，专门处理记忆整合。每隔 24 小时，或者完成 5 次会话之后，系统会 fork 一个子代理，审阅历史记录：合并相关内容、删除矛盾信息、把模糊表述固化成确定知识。

代码里这个进程的系统提示写得很文学：*"你正在做梦。反思你的记忆，合成持久知识，清理噪声。"*

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-6.png)

Auto-Dream 的设计非常工程化：它有三重门控（时间、会话数、文件级 advisory lock），并且用最便宜的检查先做筛选。锁文件本身也设计得很精巧——mtime 就是 `lastConsolidatedAt`，文件体是持有者的 PID，但即使 PID 还活着，超过一定时间也会被视为过期（防止 PID 重用攻击）。这本质上是 **Agent 记忆的垃圾回收**。

## KAIROS：还没发布的永不睡觉模式

feature flag 关着，但逻辑已写完。

**KAIROS**（古希腊语"在正确的时间"）是一个持续后台运行的代理模式：每隔几秒检查一次现在有什么值得主动做的，订阅 GitHub webhook，支持 cron 定时刷新，24 小时不间断运行。不需要你开口，它自己判断什么时候该动。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260401-7.png)

现有的 AI 编程工具都是"你叫我才动"——会话开始，任务执行，会话结束。KAIROS 如果上线，这个边界就消失了。AI 不再是工具，而是一个在后台持续工作的合作者。这是本次泄露里最重要的产品路线图信号。

## 代码里的情绪检测

当用户在对话里说出 `wtf`、`damn it`、`useless` 这类词时，系统会识别出用户处于挫败状态。识别方法是**正则表达式**，不是 LLM 推理。原因很直接：这样更快，也更省钱。

这完美体现了 Claude Code 的一个核心工程原则：**不是所有决策都需要最强模型。** 权限预检交给最便宜的 Haiku；情绪检测用正则；MicroCompact 零 API 调用。主模型只负责它真正擅长的推理与生成，其他一切——安全检查、分类、压缩、格式化——都交给最便宜且可靠的工具。

## 写在最后

这次泄露无意中成为了有史以来最全面的生产级 AI Agent 参考架构。它验证了一些我们早已讨论的概念：简单的 Agent 循环、结构化工具、纵深防御的安全、缓存感知的提示工程、怀疑论记忆。但更重要的是，它证明了**这些模式在真实的大规模生产环境中是有效的**。

对于正在构建 Agent 的团队，核心建议可以总结为：

- 从一个简单的 `while` 循环开始，而不是复杂的状态机或 DAG
- 为常见操作构建专用、类型化、权限隔离的工具，而不是暴露一个通用 Shell
- 把安全规则嵌入工具描述，用便宜模型做便宜决策
- 把上下文工程当作竞争护城河，而不是 prompt engineering
- 用轻量索引 + 按需检索设计记忆，并强制验证
- 做多代理时从第一天就考虑 **KV cache 共享**，否则并行成本会高到不可行

喧嚣会过去，但 architecture lesson 不会。六个月后，那些研究了这份代码架构而不是只看八卦的团队，会因此做出更好的 Agent。
