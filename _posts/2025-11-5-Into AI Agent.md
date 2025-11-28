---

layout: post
tags: [LLM, NLP, Agent]
title: Into AI Agent
date: 2025-11-5
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

在当今的 LLM 应用中，Agent 是一个至关重要的概念。它能帮助 LLM 完成代码生成、问题解答、多轮对话等复杂任务。然而，在众多 LLM 应用中，最成功的那些往往不依赖于复杂的架构或特殊的计算库，而是采用简单、通用的 Agent 模式。


## 什么是Agent

“Agent”可以有多种定义方式。一些客户将 Agent 定义为能够长时间独立运行、使用各种工具来完成复杂任务的完全自主的系统。另一些客户则用这个术语来描述遵循预定义 `workflow` 的、更具规定性的实现。在 Anthropic，我们将所有这些变体归类为 **agentic systems**，但在 **workflows** 和 **agents** 之间做出了一个重要的架构区分：

- **Workflows** 是指大语言模型（LLM）和工具通过预定义的代码路径被编排起来的系统。
- 另一方面，**Agents** 则是指大语言模型（LLM）动态地指导其自身流程和工具使用，并保持对任务完成方式的控制权的系统。

## 什么时候使用或者不使用 Agent

在使用 LLM 构建应用程序时，我们建议寻找尽可能简单的解决方案，仅在需要时才增加复杂性。这可能意味着根本不构建 agentic systems。Agentic systems 通常以延迟和成本换取更好的任务性能，你应该考虑这种权衡在何时是合理的。

当需要更高的复杂性时，对于定义明确的任务，workflows 提供了可预测性（predictability）和一致性（consistency），而当需要大规模的灵活性和模型驱动的决策时，agents 则是更好的选择。然而，对于许多应用程序来说，通过检索和上下文示例来优化单次 LLM 调用通常就足够了。

## 何时以及如何使用 Frameworks

当前有许多 frameworks 可以用来帮助 agentic systems 更简单地被用到，包括：

- The [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview);
- Amazon Bedrock's [AI Agent framework](https://aws.amazon.com/bedrock/agents/);
- [Rivet](https://rivet.ironcladapp.com/), a drag and drop GUI LLM workflow builder; and
- [Vellum](https://www.vellum.ai/), another GUI tool for building and testing complex workflows.

这些框架通过简化标准的底层任务（如调用 LLMs、定义和解析 tools、以及将调用链接在一起）来让起步变得容易。然而，它们常常会创造出额外的抽象层，这可能会掩盖底层的提示（prompts）和响应，从而使调试变得更加困难。它们也可能诱使你在一个更简单的设置就足够的情况下增加不必要的复杂性。

我们建议开发者从直接使用 LLM APIs 开始：许多模式只需几行代码即可实现。如果你确实使用了框架，请确保你理解其底层代码。对底层机制的错误假设是客户出错的一个常见来源。

## 搭建 blocks，workflows，以及agents

在本节中，我们将探讨我们在生产环境中看到的 agentic systems 的常见模式。我们将从我们的基础构建 blocks ——增强型 LLM——开始，并逐步增加复杂性，从简单的组合式 workflows 到自主的 agents。

### 搭建 blocks：The augmented LLM

Agentic systems 的基础构建块是一个通过增强功能（如 retrieval、tools 和 memory）得到增强的 LLM。我们目前的模型能够主动使用这些能力——生成自己的搜索查询、选择合适的工具，并决定要保留哪些信息。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-01.png)

我们建议在实现时关注两个关键方面：根据你的具体用例定制这些能力，并确保它们为你的 LLM 提供一个简单、文档齐全的接口。虽然实现这些增强功能的方法有很多，但其中一种方法是通过我们最近发布的 [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)，它允许开发者通过一个简单的客户端实现，与一个不断增长的[第三方工具生态系统集成](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients)。

在本文的其余部分，我们将假设每次 LLM 调用都能访问这些增强功能。

### Workflow: Prompt chaining

提示链 (Prompt chaining) 将一个任务分解为一系列步骤，其中每个 LLM 调用处理前一个调用的输出。你可以在任何中间步骤上添加程序化的检查（见下图中的“gate”），以确保流程仍在正轨上。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-02.png)

**何时使用此 workflow**：此 workflow 非常适用于任务可以轻松、清晰地分解为固定子任务的情况。其主要目标是通过使每个 LLM 调用成为一个更简单的任务，来用延迟换取更高的准确性。

**适用于提示链 `workflow` 场景的例子**：

- 生成营销文案，然后将其翻译成另一种语言。
- 编写文档大纲，检查大纲是否符合某些标准，然后根据大纲编写文档。

### Workflow：Routing

路由 (Routing) 对输入进行分类，并将其导向一个专门的后续任务。这个 `workflow` 允许关注点分离，并构建更专门化的 `prompts`。如果没有这个 workflow，针对一种输入的优化可能会损害在其他输入上的性能。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-03.png)

**何时使用此 workflow**：当存在可以更好地分开处理的不同类别，并且分类可以被 LLM 或更传统的分类模型/算法准确处理时，路由 (Routing) 在复杂任务中表现良好。

**适用于路由 workflow 场景的例子**：

- 将不同类型的客户服务查询（一般问题、退款请求、技术支持）引导到不同的下游流程、`prompts` 和 `tools` 中。
- 将简单/常见的问题路由到更小、更具成本效益的模型（如 Claude Haiku），并将困难/不寻常的问题路由到功能更强大的模型（如 Claude Sonnet），以优化最佳性能。

### Workflow: Parallelization

LLM 有时可以同时处理一个任务，并以编程方式聚合它们的输出。这种 workflow，即并行化 (parallelization)，主要有两种变体：

- **分片 (Sectioning)**：将一个任务分解为并行运行的独立子任务。
- **投票 (Voting)**：多次运行同一个任务以获得多样化的输出。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-04.png)

**何时使用此 `workflow`**：当划分的子任务可以为了速度而并行化时，或者当需要多个视角或尝试以获得更高置信度的结果时，并行化 (Parallelization) 是有效的。对于具有多种考虑因素的复杂任务，LLM 通常在每个考虑因素由单独的 LLM 调用处理时表现更好，从而可以集中关注每个特定方面。

**适用于并行化 workflow 场景的例子**：

**分片 (Sectioning)**：
- 实现 `guardrails`，其中一个模型实例处理用户查询，而另一个实例筛选不当内容或请求。这通常比让同一个 LLM 调用同时处理 `guardrails` 和核心响应表现得更好。
- 自动化评估 (evals) 以评估 LLM 性能，其中每个 LLM 调用评估模型在给定 `prompt` 上性能的不同方面。

**投票 (Voting)**：
- 审查一段代码的漏洞，其中几个不同的 `prompts` 会审查代码，如果发现问题就进行标记。
- 评估一段给定的内容是否不当，通过多个 `prompts` 评估不同方面或要求不同的投票阈值来平衡假阳性和假阴性。

### Workflow: Orchestrator-workers

在 orchestrator-workers (协调器-工作者) `workflow` 中，一个中心的 LLM 会动态地分解任务，将它们委托给 worker LLMs，并综合它们的结果。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-05.png)

**何时使用此 `workflow`**：此 `workflow` 非常适用于你无法预测所需子任务的复杂任务（例如，在编码中，需要更改的文件数量以及每个文件中更改的性质可能取决于任务）。虽然它在结构上与并行化相似，但关键区别在于其灵活性——子任务不是预先定义的，而是由 `orchestrator` (协调器) 根据具体输入动态决定的。

**适用于 Orchestrator-workers workflow 场景的例子**：

- 每次都对多个文件进行复杂更改的编码产品。
- 涉及从多个来源收集和分析信息以获取可能相关信息的搜索任务。

### Workflow: Evaluator-optimizer

在 evaluator-optimizer (评估器-优化器) workflow 中，一个 LLM 调用生成响应，而另一个 LLM 则在循环中提供评估和反馈。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-06.png)

**何时使用此 workflow**：当我们有明确的评估标准，并且迭代优化能提供可衡量的价值时，此 workflow 特别有效。两个合适的迹象是：首先，当人类明确表达他们的反馈时，LLM 的响应可以得到明显改善；其次，LLM 能够提供此类反馈。这类似于人类作者在撰写一篇精炼文档时可能经历的迭代写作过程。

**适用于 Evaluator-optimizer `workflow` 场景的例子**：

- 文学翻译，其中存在翻译器 LLM 最初可能无法捕捉到的细微差别，但评估器 LLM 可以提供有用的评论。
- 复杂的搜索任务，需要多轮搜索和分析以收集全面的信息，其中评估器决定是否需要进一步搜索。

## Agents

随着 LLMs 在关键能力——理解复杂输入、进行推理和规划、可靠地使用 tools、以及从错误中恢复——方面日趋成熟，Agents 正在生产环境中崭露头角。Agents 的工作始于人类用户的命令或互动式讨论。一旦任务明确，agents 就会独立规划和操作，并可能返回给人类以获取更多信息或判断。在执行过程中，至关重要的是 agents 在每一步都从环境中获取“ground truth”（例如 tool 调用结果或代码执行情况）来评估其进展。然后，Agents 可以在 checkpoints 或遇到 blockers 时暂停以获取人类反馈。任务通常在完成后终止，但包含停止条件（例如最大迭代次数）以保持控制也很常见。

`Agents` 可以处理复杂的任务，但它们的实现通常很简单。它们通常只是在循环中基于环境反馈使用 `tools` 的 LLMs。因此，清晰周到地设计 `toolsets` 及其文档至关重要。我们将在附录2 (“Prompt Engineering your Tools”) 中详细阐述工具开发的最佳实践。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-07.png)

**何时使用 agents**：Agents 可用于开放式问题，这些问题难以或不可能预测所需的步骤数，并且你无法硬编码固定的路径。LLM 可能会运行多轮，你必须对其决策有一定程度的信任。Agents 的自主性使其成为在受信任环境中扩展任务的理想选择。

`agents` 的自主性意味着更高的成本和复合错误的潜力。我们建议在沙盒环境中进行广泛测试，并配备适当的 `guardrails`。

**适用于 `agents` 场景的例子**：

以下示例来自我们自己的实现：

- 一个用于解决 SWE-bench 任务的编码 `Agent`，其中涉及根据任务描述对多个文件进行编辑；
- 我们的“计算机使用”参考实现，其中 Claude 使用计算机来完成任务。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-08.png)

## Combining and customizing these patterns

这些构建块并非规定性的。它们是开发者可以根据不同用例塑造和组合的常见模式。成功的关键，与任何 LLM 功能一样，在于衡量性能和迭代实现。重申一遍：只有在能够证明增加复杂性可以显著改善结果时，你才应该考虑这样做。

## Summary

在 LLM 领域取得成功，关键不在于构建最复杂的系统，而在于构建满足你需求的正确系统。从简单的 prompts 开始，通过全面的评估来优化它们，只有在更简单的解决方案无法满足需求时，才添加多步骤的 agentic systems。

在实现 `agents` 时，我们尝试遵循三个核心原则：

- 保持 `agent` 设计的简洁性。
- 通过明确展示 `agent` 的规划步骤来优先考虑透明度。
- 通过详尽的 `tool` 文档和测试，精心打造你的 `agent-computer interface (ACI)`。

Frameworks 可以帮助你快速入门，但在转向生产环境时，不要犹豫，减少抽象层，用基础组件来构建。通过遵循这些原则，你可以创建出不仅功能强大，而且可靠、可维护、并受用户信赖的 agents。


原文：[Building effective agents by Anthropic](https://www.anthropic.com/engineering/building-effective-agents)