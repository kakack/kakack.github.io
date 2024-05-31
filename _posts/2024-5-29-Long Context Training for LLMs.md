---

layout: post
tags: [LLM, NLP, Long Context]
title: Long Context Training for LLMs
date: 2024-05-29
author: Kaka Chen
comments: true
toc: true
pinned: false

---

# 1 - Intro

# 2 - Extrapolation and Interpolation

通常在现有的LLM模型中都会使用位置信息插入的方法，例如绝对位置嵌入（absolute positional embeddings，APE）、自学位置嵌入（learned positional embeddings，LPE，用于GPT-3、OPT模型）、相关位置嵌入（relative positional embeddings，RPE，用于Gopher、Chinchilla）、相关位置偏差（relative position bias）和旋转位置嵌入（rotary position embedding，RoPE，用于GLM、Llama系列等）。但是在有限长上下文的训练和推理中依然充满了挑战，因此基于位置信息的外推（Extrapolation）和插值（Interpolation）等技术被提出并应用。

# 3 - Recurrent Structure

# 4 - Segmentation and Sliding Window

# 5 - Memory-Retrieval Augmentation

# 6 - Conclusion