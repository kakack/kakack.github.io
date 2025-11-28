---

layout: post
tags: [LLM, NLP, Agent, RAG]
title: Context, RAG and Memory
date: 2025-11-27
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

Context、RAG、Memory 不是互斥，而是互补上下文工程用于会话即时优化，RAG用于把权威文档注入生成，长期记忆用于跨会话个性化。

## Context、RAG、Memory 对比

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/251127.png)

| **维度** | **Context Engineering**    | **RAG**                  | **Memory**               |
|:---------|:---------------------------|:-------------------------|:-------------------------|
| 本质     | 控制输入 → 激活模型内在能力 | 引入外部证据 → 抑制幻觉     | 持久化状态 → 构建个体认知   |
| 数据     | 会话内示例/摘要            | 外部文档库               | 用户历史/事件/偏好       |
| 持久性   | 临时（策略可存）           | 文档持久，检索临时       | 持久+衰减+删除           |
| 检索     | 规则/摘要压缩              | 向量+BM25+重排           | 向量+时间+标签检索       |
| 成本     | 低（仅需索引）             | 中（检索+重排）          | 高（存储+合规+维护）     |
| 延迟     | 几乎无                     | 中~高                    | 中（取决于索引）         |
| 核心价值 | 快、准、可控               | 真、可溯                 | 个性、连续、忠诚         |
| 致命风险 | 上下文窗口耗尽             | 检索出错 = 生成出错      | 错误记忆 = 信任崩塌      |
| 总结定位 | 上下文是有限有形状的容器   | 检索是显微镜             | 记忆是大脑皮层           |

简言之，context engineering 是通过控制 prompt 本身的格式和指令来让用户“说对话”，而 RAG 是通过引入外部证据来抑制幻觉，也就是“说对事”。Memory 则是通过持久化状态来构建个体认知，让用户“记得谁在说话”。

## Context、RAG、Memory 在有限资源下的取舍

## Context、RAG、Memory 三元结构