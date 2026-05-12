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

# Intro

`Harness Engineering`这个词最早是 `HashiCorp` 联合创始人 `Mitchell Hashimoto` 在 2026 年 2 月的博客中最早提出的。他的原话是：Agents 如果第一次就能给出正确的结果，或者至少只需稍作修改，效率会更高。实现这一目标最可靠的方法是为 agents 提供快速、高质量的工具，以便在出错时自动发出警报。对此，他总结为两种形式：

- 1.  **Better implicit prompting (AGENTS.md)**：对于一些简单的问题，例如代理重复运行错误的命令或找到错误的 API，请更新 AGENTS.md （或等效文件）。
- 2. **Actual, programmed tools.**： 例如，用于截屏、运行筛选测试等的脚本。通常需要修改 AGENTS.md 文件，以告知系统这些工具的存在。

每次 agents 犯错，用户都会尽力阻止它们再次犯同样的错误，或者反过来，也会尽力确保 agents 能够证明它们正在做对的事情。

---

# Reference

- [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- [Harness Engineering - first thoughts](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering-memo.html)
- [My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey)
