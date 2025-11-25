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


大模型 Agent 和 workflow 的区别在哪里？ - yuan的回答 - 知乎
https://www.zhihu.com/question/1896707093580448857/answer/1943828960061419721


https://www.anthropic.com/engineering/building-effective-agents