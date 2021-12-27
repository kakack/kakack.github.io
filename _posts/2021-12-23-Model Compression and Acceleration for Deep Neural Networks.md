---

layout: post
categories: [Algorithm]
tags: [Deep Learning, Model Compression]

---

# Abstract

随着Deep NN的广泛运用，我们发现如果每次都使用体量巨大的NN模型，会对计算资源要求非常高，因此一些关于模型压缩和加速的手段被提出来，尤其是在将模型应用到移动端的尝试，让模型应用有了如下一些优势：

- 减轻服务器端计算压力，利用云端一体化实现负载均衡；
- 实时性好，响应速度快；
- 稳定性高，可靠性好；
- 安全性高，用户隐私保护好。

其中模型的压缩和加速是两个不同的方面，它们之间有可能互相有正向影响，也有可能并无相关性，其中压缩的重点在于减少网络参数量，而加速则关注降低计算复杂性，提升并行计算能力等。总体可分为三个层次：

- 算法层压缩加速：结构优化（矩阵分解、分组卷积、小卷积核等）、量化与定点化、模型剪枝、模型蒸馏；
- 框架层加速：编译优化、缓存优化、稀疏存储和计算、NEON指令应用、算子优化等；
- 硬件层加速：AI硬件芯片层加速，如GPU、FPGA、ASIC等多种方案。

简单总结一下常被提及的四种方法：

|Theme Name|Description|Application|More Details|Drawback|
|:----:|:----:|:----:|----|----|
|参数剪枝和共享（Parameter Pruning and Sharing）|减少对性能不敏感的冗余参数|卷积层和全连接层|对不同设置有鲁棒性，能够实现好的性能，能支持脚本（Scratch）和预训练模型的训练||
||||||
||||||
||||||

# Parameter Pruning and Sharing

# Low-Rank Factorization and Sparsity

# Transfered/Compact Convolutional Filters

# Knowledge Distillation
