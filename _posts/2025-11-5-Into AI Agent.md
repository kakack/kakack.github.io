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

在当前工业界和学术界的 LLM 应用中，Agent 是一种非常重要的组件。它可以帮助 LLM 完成复杂的任务，例如自动生成代码、解决问题、进行对话等。然而在这错综复杂品类繁多的 LLM 应用中，那些最成功的应用往往不是那些使用非常复杂架构或者非常特殊计算库的，反而都是那些简单的、通用模式的 Agent。


# 什么是Agent

“Agent”可以有多种定义方式。一些客户将 Agent 定义为能够长时间独立运行、使用各种工具来完成复杂任务的完全自主的系统。另一些客户则用这个术语来描述遵循预定义 workflow 的、更具规定性的实现。在 Anthropic，我们将所有这些变体归类为 **agentic systems**，但在 **workflows** 和 **agents** 之间做出了一个重要的架构区分：

- **Workflows** 是指大语言模型（LLM）和工具通过预定义的代码路径被编排起来的系统。
- 另一方面，**Agents** 则是指大语言模型（LLM）动态地指导其自身流程和工具使用，并保持对任务完成方式的控制权的系统。

# 什么时候使用或者不使用 Agent

在使用 LLM 构建应用程序时，我们建议寻找尽可能简单的解决方案，仅在需要时才增加复杂性。这可能意味着根本不构建 agentic systems。Agentic systems 通常以延迟和成本换取更好的任务性能，你应该考虑这种权衡在何时是合理的。

当需要更高的复杂性时，对于定义明确的任务，workflows 提供了可预测性（predictability）和一致性（consistency），而当需要大规模的灵活性和模型驱动的决策时，agents 则是更好的选择。然而，对于许多应用程序来说，通过检索和上下文示例来优化单次 LLM 调用通常就足够了。

# 什么时候如何使用 framworks

当前有许多 frameworks 可以用来帮助 agentic systems 更简单地被用到，包括：

- The [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview);
- Amazon Bedrock's [AI Agent framework](https://aws.amazon.com/bedrock/agents/);
- [Rivet](https://rivet.ironcladapp.com/), a drag and drop GUI LLM workflow builder; and
- [Vellum](https://www.vellum.ai/), another GUI tool for building and testing complex workflows.

这些框架通过简化标准的底层任务（如调用 LLMs、定义和解析 tools、以及将调用链接在一起）来让起步变得容易。然而，它们常常会创造出额外的抽象层，这可能会掩盖底层的提示（prompts）和响应，从而使调试变得更加困难。它们也可能诱使你在一个更简单的设置就足够的情况下增加不必要的复杂性。

我们建议开发者从直接使用 LLM APIs 开始：许多模式只需几行代码即可实现。如果你确实使用了框架，请确保你理解其底层代码。对底层机制的错误假设是客户出错的一个常见来源。

# 搭建 blocks，workflows，以及agents

在本节中，我们将探讨我们在生产环境中看到的 agentic systems 的常见模式。我们将从我们的基础构建 blocks ——增强型 LLM——开始，并逐步增加复杂性，从简单的组合式 workflows 到自主的 agents。

## 搭建 blocks：The augmented LLM

Agentic systems 的基础构建块是一个通过增强功能（如 retrieval、tools 和 memory）得到增强的 LLM。我们目前的模型能够主动使用这些能力——生成自己的搜索查询、选择合适的工具，并决定要保留哪些信息。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-01.png)

我们建议在实现时关注两个关键方面：根据你的具体用例定制这些能力，并确保它们为你的 LLM 提供一个简单、文档齐全的接口。虽然实现这些增强功能的方法有很多，但其中一种方法是通过我们最近发布的 [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)，它允许开发者通过一个简单的客户端实现，与一个不断增长的[第三方工具生态系统集成](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients)。

在本文的其余部分，我们将假设每次 LLM 调用都能访问这些增强功能。

## Workflow: Prompt chaining

提示链 (Prompt chaining) 将一个任务分解为一系列步骤，其中每个 LLM 调用处理前一个调用的输出。你可以在任何中间步骤上添加程序化的检查（见下图中的“gate”），以确保流程仍在正轨上。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-02.png)

**何时使用此 workflow**：此 workflow 非常适用于任务可以轻松、清晰地分解为固定子任务的情况。其主要目标是通过使每个 LLM 调用成为一个更简单的任务，来用延迟换取更高的准确性。

**适用于提示链 workflow 场景的例子**：

- 生成营销文案，然后将其翻译成另一种语言。
- 编写文档大纲，检查大纲是否符合某些标准，然后根据大纲编写文档。

## Workflow：Routing

路由 (Routing) 对输入进行分类，并将其导向一个专门的后续任务。这个 workflow 允许关注点分离，并构建更专门化的 prompts。如果没有这个 workflow，针对一种输入的优化可能会损害在其他输入上的性能。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251105-03.png)

**何时使用此 workflow**：当存在可以更好地分开处理的不同类别，并且分类可以被 LLM 或更传统的分类模型/算法准确处理时，路由 (Routing) 在复杂任务中表现良好。

**适用于路由 workflow 场景的例子**：

- 将不同类型的客户服务查询（一般问题、退款请求、技术支持）引导到不同的下游流程、prompts 和工具中。
- 将简单/常见的问题路由到更小、更具成本效益的模型（如 Claude Haiku），并将困难/不寻常的问题路由到功能更强大的模型（如 Claude Sonnet），以优化最佳性能。

## Workflow: Parallelization

LLM 有时可以同时处理一个任务，并以编程方式聚合它们的输出。这种 workflow，即并行化 (parallelization)，主要有两种变体：

- **分片 (Sectioning)**：将一个任务分解为并行运行的独立子任务。
- **投票 (Voting)**：多次运行同一个任务以获得多样化的输出。

**何时使用此 workflow**：当划分的_子任务可以为了速度而并行化时，或者当需要多个视角或尝试以获得更高置信度的结果时，并行化 (Parallelization) 是有效的。对于具有多种考虑因素的复杂任务，LLM 通常在每个考虑因素由单独的 LLM 调用处理时表现更好，从而可以集中关注每个特定方面。

**并行化有用的例子**：

**分片 (Sectioning)**：
- 实现 `guardrails`，其中一个模型实例处理用户查询，而另一个实例筛选不当内容或请求。这通常比让同一个 LLM 调用同时处理 `guardrails` 和核心响应表现得更好。
- 自动化评估 (evals) 以评估 LLM 性能，其中每个 LLM 调用评估模型在给定 `prompt` 上性能的不同方面。

**投票 (Voting)**：
- 审查一段代码的漏洞，其中几个不同的 `prompts` 会审查代码，如果发现问题就进行标记。
- 评估一段给定的内容是否不当，通过多个 `prompts` 评估不同方面或要求不同的投票阈值来平衡假阳性和假阴性。





大模型 Agent 和 workflow 的区别在哪里？ - yuan的回答 - 知乎
https://www.zhihu.com/question/1896707093580448857/answer/1943828960061419721


https://www.anthropic.com/engineering/building-effective-agents