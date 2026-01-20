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