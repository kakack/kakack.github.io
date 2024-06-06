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

位置外推扩展（positional extrapolation）的策略将位置信息的编码扩展到模型在训练期间明确学习到的内容之外。比如**ALiBi**应用线性偏差注意力来对超过训练期间看到的最大长度的序列进行外推。通过应用负偏注意力分数，并根据相关键和查询之间的距离线性减少惩罚，而不是使用位置嵌入，它可以促进有效的长度外推。简单来说就是给注意力加上线性偏置的方法，来处理“训练的序列长度可以开到2k，而推理的序列长度可以开到4k”的情况。在计算每一个head的attention score时，它用一个和query、key之间的距离成比例的一个“惩罚项”来偏置query-key的注意力得分。整体公式可以理解成：

$$softmax(q_iK^T+m [-(i-1),...,-2,-1,0])$$

其中第一项依然是常规transformer里的Q和K内积，第二项是一个基于相对距离的负数作为惩罚值，m是一个常数值。

此外也有XPOS将注意力解决度作为外推的标记，并利用相对位置嵌入来增强注意力解决度，从而改善长度外推。CLEX 建议用常微分方程来推广位置嵌入缩放，以对长度缩放因子的连续动态进行建模。通过这样做，CLEX 摆脱了现有位置外推缩放方法的限制，从而能够生成长序列。

位置插值（positional interpolation）策略会减少输入位置索引的规模并扩展上下文窗口大小，从而使 LLM 能够在较长的文本序列上保持其性能，超出训练的上下文长度可能会损害自注意力机制。常见的RoPE基础上，NTK interpolation改变了RoPE的基数，从而有效地改变了每个 RoPE 维度的旋转速度。YaRN 插值使用斜坡函数以不同比例在各个维度上混合线性和 NTK 插值，并结合温度因子来抵消由于长输入导致的注意力矩阵中的分布偏移。 FIRE提出了一种功能性相对位置编码，使用可学习的输入位置到偏差的映射和渐进插值，确保编码函数在所有序列长度上的输入有界，从而实现长度泛化。PoSE提出了位置跳跃式训练，它使用固定上下文窗口智能地模拟长输入，并设计不同的跳跃偏差项来操纵每个块的位置索引。与全长微调相比，此策略减少了内存和时间开销。

# 3 - Recurrent Structure

LLM 管理长序列的能力也可以通过递归结构Recurrent Structure得到增强。例如，Transformer-XL提出了一种段级递归机制，并利用增强的相对位置编码来捕获长期依赖关系并解决长期上下文碎片化问题。Memformer利用外部动态存储器来编码和检索过去的信息，实现了长序列的线性时间和恒定的内存空间复杂度。它还提出了记忆重放反向传播Memory Replay Back-Propagation (MRBP)，以促进随时间变化的长距离反向传播，同时显着降低内存要求。∞-former提出了一种增强了无界长期记忆unbounded long-term memory(LTM) 的 Transformer 模型，采用连续空间注意框架来平衡内存中容纳的信息单元数量与其表示的粒度。Recurrent Memory Transformer(RMT) 使用循环机制，通过将特殊记忆标记合并到输入或输出序列中来保留过去段级别的信息，与Transformer-XL 相比，在长上下文建模中表现出色。Block-Recurrent Transformers利用自注意力和交叉注意力在广泛的状态向量和标记集上执行循环函数，从而通过并行计算对长序列进行建模。最后，保留网络Retentive Network引入了一种多尺度保留机制，作为多头注意力的替代方案。通过包含并行和块级循环表示，它可以实现有效的扩展，允许并行训练，并实现训练并行化和恒定推理成本，同时与其他Transformer模型相比，提供线性长序列记忆复杂度。

# 4 - Segmentation and Sliding Window

分段Segmentation和滑动窗口Sliding Window技术通过将输入数据划分为较小的段或应用移动窗口来滑动长序列来解决长上下文处理的问题。例如Mistral使用滑动窗口注意力来有效处理任意长度的序列，同时降低推理成本。StreamingLLM发现了一种注意力下沉现象，并指出保留初始标记的键值可显著恢复窗口注意力的性能。基于这一观察，它通过合并窗口上下文和第一个标记提出了一个有效的框架，允许使用有限长度注意力窗口训练的 LLM，但能够推广到无限序列长度而无需任何微调。并行上下文窗口Parallel Context Windows (PCW)将长上下文分段为块，将注意力机制限制为仅在每个窗口内起作用，然后在这些窗口之间重新部署位置嵌入。LongNet提出了扩张注意力机制，随着距离的增加，注意力范围呈指数级扩大，从而能够处理长度超过10亿个token的序列。LongNet可以通过对序列维度进行分区来实现并行训练。SLED是一种处理长序列的简单方法，它重新利用并利用经过充分验证的短文本语言模型用于LLM。

# 5 - Memory-Retrieval Augmentation

一些研究通过使用记忆检索增强memory-retrieval augmentation策略来解决极长文本的推理问题。一个值得注意的例子是KNN-augmented Transformer，它通过利用k-nearest-neighbor(KNN)查找来获取以前相似的上下文嵌入，从而扩展了注意力上下文的大小。Landmark Attention使用landmark token来表示每个输入块，并训练注意力机制利用它来选择相关块。这允许通过注意力机制直接检索块，同时保持先前上下文的随机访问灵活性，在LLaMA-1上展示了令人印象深刻的长上下文建模性能。LongMem提出了一种解耦网络架构，其中原始骨干LLM作为记忆编码器，自适应残差侧网络adaptive residual side networ作为记忆检索器和读取器，有效地缓存和更新长期过去的上下文以防止知识陈旧。Unlimi-former通过将注意力点积得分输出为KNN距离来增强KNN-augmented Transformer，从而实现对几乎无限的输入序列的索引。Focused Transformer (FoT)强调，随着上下文长度的增加，相关键与不相关键的比例会减小，并提出了一种通过对比学习来优化键值空间结构的优化解决方案。最后研究发现，在生成过程中使用简单检索进行增强时，具有4K上下文窗口的LLM可以在长上下文任务上使用位置插值匹配具有16K上下文窗口的微调LLM的性能，同时需要的计算量要少得多。

# 6 - Implement

## 6.1 - Sequence parallelism

当前所说的序列并行Sequence Parallelism可以概括为两类,分别是:
- Colossal-AI提出的 `Sequence Parallelism: Long Sequence Training from System Perspective`，主要是解决模型的输入长度(sequence length)限制；
- Megatron-LM提出的 `Reducing Activation Recomputation in Large Transformer Models`，主要是减少模型显存。

Colossal-AI的SP诞生的背景是 self-attention 的内存需求是输入长度（sequence length）的2次方。因此需要有一种内存高效的并行方法，可以破输入序列长度限制，并在 GPU 上有效地训练更长的序列，同时与大多数现有的并行技术兼容。具体而言是将整个输入序列分块chunks，并将每个chunk输入到相应设备（GPU），通过ring-allreduce形式与self-attention计算结合，得到一个Ring Self-Attention（RSA）形式。

Megatron-LM提出的SP是为了通过其他方式分摊Tensor Parallelism中无法分摊的显存，将 Transformer层中的LayerNorm以及Dropout的输入按输入长度（Sequence Length）维度进行了切分，使得各个设备上面只需要做一部分的 Dropout 和 LayerNorm 即可。

## 6.2 - DeepSpeed ZeRO3 Offload

## 6.3 - Flash Attention & Fused Cross Entropy Kernel

## 6.4 - Activation Checkpointing

# 6 - Conclusion