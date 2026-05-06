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

近年来，主流大语言模型架构正经历从`标准多头注意力（MHA）`向`多查询注意力（MQA）`、`分组查询注意力（GQA）`及`多头潜在注意力（MLA）`的范式转移。这一演进的核心驱动力在于解决自回归解码阶段 KV Cache 带来的显存容量与带宽瓶颈（即“内存墙”问题），旨在通过降低访存开销来显著提升推理吞吐量与长文本处理能力。
 
当前 transformer 的组件中，尤其是 attention 部分从硬件适配和自身结构上有多方面的优化。如 Qwen3-Next、Llama 的 GQA，DeepSeek 的 MLA，minimax-M2 的 MHA 等。那这些优化到底是出于何种目的？想要回答这个问题，需要先从推理的两个阶段说起。

## 一、推理的两个阶段：Prefill 与 Decode

LLM 在 inference 阶段可以拆解为两个性质截然不同的步骤：

- **Prefill 阶段**：模型接收完整的输入 Prompt，处于一个`预填充状态`。**并行**计算所有输入词元的中间表示（$Q$、$K$、$V$），并生成第一个输出词元。该过程是矩阵-矩阵乘法（GEMM），计算高度密集，能充分利用 GPU 的并行计算单元，属于典型的 **计算受限（Compute-bound）** 场景。
- **Decode 阶段**：从第二个 token 开始，模型每次只生成一个新 token，并需要让当前 token 的 $\vec{q}$ 与历史序列的 $K$、$V$ 做注意力交互。如果不做任何缓存，每生成一个 token 都要为所有历史词元重新计算一遍 $K$、$V$，复杂度会随序列长度平方级膨胀，造成大量重复计算，理论复杂度攀升到$O(N^2)$。

正是 Decode 阶段“自回归 + 重复计算”的特性，催生了 **KV Cache** 这一“以空间换时间”的优化思路。

## 二、MHA 的计算流程

### 2.1 Prefill 阶段

模型接收到完整的输入序列，可以写作: $X=[\vec{x_1}, \vec{x_2}, ... ,\vec{x_t},]$
在 MHA 的 prefill 阶段：

1. 输入序列经过线性层$W_q, W_k, W_v$投影成 $Q、K、V$，$Q,K,V \in \mathbb{R}^{t \times d}$， 并按 head 切分，维度变化为 $[b, h, L, d_h]$；
2. 通过 $Q \times K^T$ 得到 $[b, h, L, L]$ 的注意力分数矩阵，再经 mask、softmax，与 $V$ 相乘聚合；
3. 多头结果拼接还原回 $[b, L, d]$，依次经过 `Add & Norm`、`FFN` 等模块；
4. 最终只取输出矩阵的最后一行 $[b, 1, d]$ 进入 `LM Head` 用于预测下一个 token，同时把整段序列对应的 $K$、$V$ 写入 **KV Cache**，作为后续增量计算的基础。

在一个 multi-head attn 下，我们的 head 为 2，则会出现以下过程：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260116-kvcache-01.png)

我们每个head，都要执行一次attention的score计算：

$$
\mathrm{Scores}_1 = Q_1 K_1^T =
\underbrace{
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1 \\
0 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}
}_{Q_1(6 \times 2)}
\times
\underbrace{
\begin{bmatrix}
2 & 0 & 2 & 0 & 2 & 0 \\
0 & 2 & 2 & 0 & 0 & 2
\end{bmatrix}
}_{K_1^T(2 \times 6)}
=
\begin{bmatrix}
2 & 0 & 2 & 0 & 2 & 0 \\
0 & 2 & 2 & 0 & 0 & 2 \\
2 & 2 & 4 & 0 & 2 & 2 \\
0 & 0 & 0 & 0 & 0 & 0 \\
2 & 0 & 2 & 0 & 2 & 0 \\
0 & 2 & 2 & 0 & 0 & 2
\end{bmatrix}
$$

在 Transformer 结构中，每一层的输出向量（Hidden State）中，只有最后一行（即最新 Token 的向量）会被传递到最终的LM Head或下一层对应的位置 ，用于生成下一个预测结果。

在 Prefill 阶段因为我们没有见过所有输入，所以需要并行计算整个 attn 矩阵取最后一行进入 LM Head 预测下一个 token。然而在 Decoding 阶段，当我们已经有一个 token 生成后，需要计算下一个 token，当生成第 $t$ 个 token时，我们需要$[x_1, ... , x_{t-1}]$的 $K$ 和 $V$ 来计算注意力，但这些历史 token 的 $K$ 和 $V$ 在生成第 $t-1$ 个 token 的时候已经算过一遍了。

### 2.2 Decode 阶段：有无 KV Cache 的对比

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260116-kvcache-02.png)

上图中所有浅色部分都是计算过一遍的，随着 context length 的增长，计算资源被大量浪费在重复计算上。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260116-kvcache-03.png)

- **没有 KV Cache**：每生成一个新 token，都需要把已经生成过的所有 token 重新跑一遍 attention，$K$、$V$ 被反复计算，浪费严重。生成 $N$ 个 token 的总复杂度量级直接攀升到 $O(N^2)$。
- **有 KV Cache**：把历史 token 的 $K$、$V$ 缓存起来，每一步只为当前新 token 计算一行 $q$、$k$、$v$，再把新的 $k$、$v$ 拼接到缓存上。$Q$ 与缓存的完整 $K$、$V$ 做注意力计算，由 **GEMM 退化为 GEMV**，计算量大幅下降。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260116-kvcache-04.png)

这样，计算第 $t$ 个 token 的输出就可以忽略 $[\vec{x_1}, ... , \vec{x_{t-1}}]$ 所在的历史上下文，而 $\vec{x_t}$ 是唯一需要的心输入，这样避免了对历史 token 进行重复的 Embedding 和线性层计算。接着用权重矩阵 $W_q, W_k, W_v$ 和当前的 $\vec{x_t}$ 计算对应向量：

$$
\vec{q_t}=\vec{x_t}W_q \\
\vec{k_t}=\vec{x_t}W_k \\
\vec{v_t}=\vec{x_t}W_v \\
$$

此时生成的 $\vec{q_t}, \vec{k_t}, \vec{v_t}$ 都只是行向量而不是大矩阵，因此很明显由 `矩阵-矩阵乘法（GEMM）` 退化为了 `举证-向量乘法（GEMV）`，计算量大大减小。当前的 query vector $\vec{q_t}$ 与完整的 $K_{new}$ 计算 attn score，并作用于：

$$
V_{new}=Softmax(\vec{q_t}K^T_{new})V_{new}
$$

将最新的 $\vec{k_t}, \vec{v_t}$ 追加写入现存中的 cache 区域，为生成第 $t+1$ 个 token 做好准备，这就是 `KV Cache`。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260116-kvcache-05.png)

> 这也顺带回答了一个常见疑问：**为什么没有 Q Cache？** 因为 decode 阶段每一步真正参与计算的只有当前 token 对应的那一行 $q$ 向量，历史 $Q$ 的信息在后续步骤中不再被使用，自然无需缓存。

## 三、KV Cache 带来的新问题：从“救星”到“瓶颈”

KV Cache 的本质是 **空间换时间**。它确实节省了计算量，却把压力转嫁到了显存上。

### 3.1 显存占用

每个 token 需要缓存的显存量近似为：

$$
\text{Mem} = 2 \times n_\text{layers} \times n_\text{heads} \times d_\text{head} \times P_\text{precision}
$$

其中：

- `2`：K 和 V 两个矩阵；
- `n_layers`：模型层数；
- `n_heads × d_head`：等价于隐藏维度 `d_model`；
- `P_precision`：数值精度（FP16 = 2 bytes，FP32 = 4 bytes）。

以 Llama-2-7B（标准 MHA）为例，`n_layers = 32`、`d_model = 4096`。当上下文长度达到 4K 时，单个序列的 KV Cache 已经占用 GB 级显存；一旦推理并发上来（例如 batch size = 32 甚至更大），即使是 80GB 的 A100 也会迅速被 KV Cache 吃满。

### 3.2 显存带宽：真正的“内存墙”

比显存容量更棘手的，是 **显存带宽**。GPU 中 HBM（显存）和 SRAM（计算单元）是两个独立的物理结构，attention 的实际计算发生在 SRAM 中，而缓存的 $K$、$V$ 存放在 HBM 中。每一步 decode 都需要把 KV Cache 从 HBM 搬运到 SRAM，搬运的速度由显存带宽决定。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260116-kvcache-06.png)


| 架构 | 显卡 | 显存带宽 | FP16/BF16 算力 | 显存类型 |
| --- | --- | --- | --- | --- |
| Volta | Tesla V100 | 900 GB/s | ~125 TFLOPS | HBM2 |
| Turing | Titan RTX | 672 GB/s | ~130 TFLOPS | GDDR6 |
| Ampere（计算卡） | A100 | 2039 GB/s | 312 TFLOPS | HBM2e |
| Ampere（游戏卡） | RTX 3090 Ti | 1008 GB/s | ~160 TFLOPS | GDDR6X |
| Hopper | H100（SXM） | 3352 GB/s | 989 TFLOPS | HBM3 |
| Ada Lovelace | RTX 4090 | 1008 GB/s | ~330 TFLOPS | GDDR6X |


以 Llama-7B（FP16）为例，权重约 14GB，假设 KV Cache 累积到 1GB，每生成一个 token 就要搬运约 15GB 数据。在 A100（带宽约 2 TB/s，算力 312 TFLOPS）上：

- 搬运耗时: 15 GB/2000 GB/s = 7.5 ms
- 计算耗时: 7B 模型大约对应 14G FLOPS。A100 算力 312 TFLOPS。计算时间 ≈ 0.04 ms
- **搬运 / 计算 ≈ 187.5 倍**

也就是说，绝大部分时间 GPU 的计算单元都在“摸鱼”等数据。这正是所谓的 **内存墙（Memory Wall）**：decode 阶段是典型的 **访存受限（Memory-bound）** 场景，而非计算受限。可以通过下列公式计算：

$$
\text{Latency} \approx \frac{\text{Model Weights} + \text{KV Cache Size}}{\text{Memory Bandwidth}}
$$

### 3.3 优化方向：从公式里能动什么？

回看 KV Cache 的显存公式：

$$
Size_{token} = 2 \times n_{layers} \times n_{heads} \times d_{heads} \times P_{precision}
$$

逐项排除：

- 2：自注意力机制的根基，不能动；
- $n_{layers}$：决定模型深度与抽象能力，动了容易“降智”；
- $d_{head}$：通常为 128，决定每个头的特征容量，砍了同样伤模型质量；
- $P_{precision}$：可以走低比特量化（INT8 / INT4），属于另一条正交的优化路线，可叠加使用；
- $n_{heads}$：相对“可动”的维度，尤其是其中的 **KV Head 数量**。

由此自然引出了核心问题：**我们真的需要那么多 KV Head 吗？** 这就是 MQA、GQA、MLA 出现的根本动机。

## 四、MQA / GQA / MLA：从架构层“瘦身”KV Cache

通过减少需要缓存的键值头数（`num_key_value_heads`），可以直接缩小 KV Cache 的体积，从而缓解显存带宽压力。

**MQA（Multi-Query Attention）**：所有 Query Head 共享 **同一组** Key/Value Head，KV Cache 大小相对 MHA 直接减少 `n_heads` 倍。代价是模型质量通常会出现可感知下降。其直接结果就是 $H_{kv}=1$。

**GQA（Grouped-Query Attention）**：将 Query Head 分成若干组，每组共享一组 KV Head。Group 数即最终的 KV Head 数，是介于 MHA 与 MQA 之间的折中方案，可在推理速度和模型质量之间灵活权衡。Llama-2/3、Qwen 系列等均采用 GQA。这里的 Group 数就是最终 KV Head 的数量，即 $H_{kv}=g$ ，每个 KV Head 负责服务的 Query Head 数量为 $\frac{H_{query}}{H_{kv}}$ 

**MLA（Multi-head Latent Attention）**：DeepSeek 提出的另一条思路，**不直接减少 KV Head 数量**，而是通过低秩潜在表示在计算图层面重构 Attention 结构，使得真正需要缓存到 HBM 的张量被显著压缩，同时尽量保持模型表达能力。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/260116-kvcache-07.png)

三者目标一致——**压缩 KV Cache、突破内存墙**——但路径不同：MQA/GQA 走“物理瘦身”，MLA 走“结构重构”。具体的实现细节、各自的优缺点与工程难点，将在后续篇章中展开。

## 五、小结

- Prefill 是计算受限场景，Decode 是访存受限场景；
- KV Cache 用空间换时间，解决了 Decode 阶段的重复计算问题；
- 但 KV Cache 自身又带来了显存容量和带宽两个新瓶颈，构成 LLM 推理的“内存墙”；
- MHA → MQA / GQA / MLA 的演进，本质上都是围绕“如何让 KV Cache 更小、更快地被读出”展开的架构层优化。

理解了 KV Cache 与内存墙，才能真正看懂当前主流 LLM 在 Attention 模块上五花八门选择背后的统一逻辑。

## 参考

1. Ainslie, J., et al. (2023). *GQA: Training generalized multi-query transformer models from multi-head checkpoints.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
2. Shazeer, N. (2019). *Fast transformer decoding: One write-head is all you need.* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
3. Hugging Face. *Optimizing inference.* [Transformers Documentation](https://huggingface.co/docs/transformers/llm_optims)
4. NVIDIA. *NVIDIA Ada Lovelace Architecture White Paper.* [PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
5. 苏剑林. *缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA.* [Science Spaces](https://spaces.ac.cn/archives/10091)
6. 原文参考：[KV Cache（一）：从 KV Cache 看懂 Attention（MHA、MQA、GQA、MLA）的优化](https://mp.weixin.qq.com/s/C6fDzijgDNNjipq5d5HGOQ)
