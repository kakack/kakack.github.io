---

layout: post
tags: [LLM, NLP, AI Infra]
title: KV Cache 如何影响了 LLM Inference
date: 2026-1-16
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

近年来，主流大语言模型架构正经历从标准多头注意力（MHA）向多查询注意力（MQA）、分组查询注意力（GQA）及多头潜在注意力（MLA）的范式转移。这一演进的核心驱动力在于解决自回归解码阶段 KV Cache 带来的显存容量与带宽瓶颈（即“内存墙”问题），旨在通过降低访存开销来显著提升推理吞吐量与长文本处理能力。

当前 transformer 的组件中，尤其是 attention 部分从硬件适配和自身结构上有多方面的优化。如 Qwen3-Next、Llama 的 GQA，DeepSeek 的 MLA，minimax-M2 的 MHA 等。那这些优化到底是出于何种目的？

首先简单赘述一下 LLM 在 inference 阶段的两个核心步骤：`prefill`和`decode`。

`prefill`阶段，模型接收完整的输入序列，并行计算输入词元的

`decode`阶段，模型接收部分输入序列，通过自回归解码生成输出序列。在这个过程中，模型需要对每个位置的 token 进行注意力计算，以理解输入序列的上下文。