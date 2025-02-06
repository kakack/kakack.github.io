---

layout: post
tags: [LLM, NLP]
title: 推测解码Speculative Decoding
date: 2024-10-10
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

推测解码（Speculative Decoding）是一种大模型的推理加速方式。
传统的解码方式需要逐步调用大模型自回归地预测每个单词，耗时较长。推测解码使用某种方法，先生成一系列草稿tokens，将这些草稿作为候选序列传入大模型，然后由大模型在单次前向传播中验证这些候选序列是否符合其高质量生成标准。因此，推测解码分为 **“草稿draft”** 和 **“验证velify”** 两个串行阶段。

这使得大模型避免了逐个token推理的过程，大幅减少计算时间，同时保留了生成的连贯性和准确性。如果利用得当，推测解码可以在不牺牲大模型生成质量的情况下，大幅增加大模型的解码速度。

# 分类

目前的推测解码（Speculative Decoding）方法可以大致分为3类：
- 在内存中同时加载一个小草稿模型，辅助生成草稿（最初的 Speculative Decoding 论文提出的方法）。这个小草稿模型可以是同系列的参数更小的模型，也可能是额外训练的轻量级模块；
- 直接将大模型同时作为起草模型和验证模型。这种方法往往会对原始模型结构做一些修改，或使用额外数据再微调大模型；
- 从外部数据中（如外部数据库或 prompt）检索获取草稿来源。

# 小模型draft+大模型verify

推测解码的核心思想在于充分利用大模型的 logits 层输出信息，而不仅仅是最后一个输出的 logits 向量的信息，在单次前向传播中实现对多个候选后续序列的验证。

其基本方法是：

1. Drafting/草稿阶段：使用某种方法（小草稿模型或大模型本身），对下一个可能出现的一批 token 进行预生成。也就是说，在大模型自回归生成一个 token 之前，先行快速给出一批候选 token 序列。
2. Validation/验证阶段：将这批候选 token 序列连同原有上下文输入大模型，通过禁用缓存或重新传入完整序列等方式，使大模型在一次前向计算中输出相应位置上的 logits 分布。在这里，大模型能通过单次前向传播就给出整个草稿序列的各位置下一步 token 的概率向量。通过将大模型针对这些位置所得的概率分布与草稿序列进行对比，可以快速判断每个草稿 token 是否与大模型的真实预测概率匹配。

---

# Reference

- [Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation](https://arxiv.org/abs/2203.16487)
- [Fast Inference from Transformers via Speculative Decoding, Yaniv Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling, Charlie Chen et al., 2023](https://arxiv.org/abs/2302.01318)

