---

layout: post
tags: [Llama2, LLM, NLP]
title: Inside Llama2
date: 2024-04-11
author: Kaka Chen
comments: true
toc: true
pinned: false

---
# 1 - Intro

Meta的Llama2是当前开源状态最好又可以作为效果标杆的一个LLM模型，但它的官方口径好像也是个半开源，即只有inference而没有train，但是从它的模型结构和部分处理逻辑上，还是具有很高的参考价值。

- [Paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2307.09288)
- [Code](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/llama)
- [Checkpoint](https://huggingface.co/meta-llama)

# 2 - Process

关于通用的LLM对于文本的处理一般是以下流程：

**输入数据**：LLM的输入数据是一段文本，可以是一个句子或一段话。文本通常被表示成单词或字符的序列。

`[君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。...五花马、千金裘，呼儿将出换美酒，与尔同销万古愁]
`

**Tokenization**：之后需要将文本进行Tokenization，将其切分成单词或字符，形成Token序列。之后再将文本映射成模型可理解的输入形式，将文本序列转换为整数索引序列(这个索引就是单词或字符在语料库中的index)，这个过程通常由一些开源的文本Tokenzier工具，如sentencepiece等来处理

序列化-> 
`['BOS','君','不','见','黄','河','之','水','天','上','来','，' ,'奔','流','到'...'与','尔','同','销','万','古','愁','EOS']`

假设语料库索引化->
`['BOS','10','3','67','89','21','45','55','61','4','324','565' ,'789','6567','786'...'7869','9','3452','563','56','66','77','EOS']`

**Embedding**：文本信息经过Tokenization之后变成了token序列，而Embedding则继续将每个Token映射为一个实数向量，为Embeding Vector。

```text
'BOS'-> [p_{00},p_{01},p_{02},...,p_{0d-1}]
'10' -> [p_{10},p_{11},p_{12},...,p_{1d-1}]
'3'  -> [p_{20},p_{21},p_{22},...,p_{2d-1}]
...
'EOS'-> [p_{n0},p_{n1},p_{n2},...,p_{nd-1}]
```

**位置编码**：对于Token序列中的每个位置，添加位置编码（Positional Encoding）向量，以提供关于Token在序列中位置的信息。位置编码是为了区分不同位置的Token，并为模型提供上下文关系的信息。




# 3 - Architcture

## 3.1 - RMSNorm

## 3.2 - RoPE

## 3.3 - KV Cache

## 3.4 - Group Query Attention

## 3.5 - FeedForward

# 4 - Training