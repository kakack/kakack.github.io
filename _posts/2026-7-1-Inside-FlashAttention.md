---

layout: post
tags: [LLM, NLP, AI Infra, Attention]
title: FlashAttention：从 IO 感知到异步流水线
date: 2026-7-1
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

几乎所有现代 LLM 的训练与推理都默认开着 FlashAttention，以至于它常常被当成一个「开了就快、开了就省显存」的黑盒开关。但如果你需要训练超长上下文、给模型加一种新的 attention 变体、或者把它移植到一块新硬件上，就绕不开它内部到底做了什么。

这篇文章想把 FlashAttention 讲透：它解决的是什么瓶颈、核心算法（tiling + online softmax + recomputation）怎么推导、从 FA1 到 FA2 再到 FA3 每一代具体改了什么，以及在不同模型结构上做定制化开发时，应该从「接口层」还是「kernel 层」入手。

---

## 一、为什么需要 FlashAttention

### 1.1 标准 attention 的两个代价

标准的 scaled dot-product attention 是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

其中 $Q, K, V \in \mathbb{R}^{S \times d}$，$S$ 是 sequence length，$d$ 是 head dimension。朴素实现会分三步走：

```python
S_ = (Q @ K.transpose(-1, -2)) / math.sqrt(d)   # [S, S]，显式 materialize
P = torch.softmax(S_, dim=-1)                    # [S, S]
O = P @ V                                         # [S, d]
```

问题就出在中间那个 $[S, S]$ 的注意力矩阵上：

- **显存代价**：它的大小是 $O(S^2)$。当 $S = 8K$ 时，单个 head 的 attention score 在 FP16 下就是 $8192^2 \times 2 \approx 128\text{MB}$；乘上 batch、head、层数，长序列下这是灾难性的。
- **读写代价**：$S_{\_}$ 和 $P$ 都要写回 GPU 的高带宽显存（HBM）再读回来。这一来一回的 HBM 读写，才是真正的瓶颈。

### 1.2 attention 是 memory-bound，不是 compute-bound

这是理解 FlashAttention 的前提。GPU 的内存层级大致是：

| 层级 | 容量（量级） | 带宽（量级） |
|---|---|---|
| 寄存器 / SRAM（on-chip） | 每个 SM ~100KB 级 | ~19 TB/s（A100） |
| HBM（显存） | 几十 GB | ~1.5–3 TB/s |

SRAM 比 HBM 快一个数量级，但小得多。标准 attention 把整个 $[S,S]$ 矩阵在 HBM 上写出来又读回去，绝大部分时间花在等内存搬运，而不是 Tensor Core 算 matmul——也就是说它是 **memory-bound** 的。

FlashAttention 的核心洞见就是：**不要把 $S \times S$ 写进 HBM**。如果能把 attention 拆成小块，让每块都在 SRAM 里算完、只把最终的 $[S, d]$ 输出写回 HBM，就能把 HBM 读写量从 $O(S^2)$ 降到接近 $O(S)$，从而把这个 memory-bound 的操作做到接近 compute-bound 的效率。

难点在于：softmax 需要对一整行做归一化（要知道这一行的最大值和指数和），而分块计算时，我们一次只看到一行的一小段。怎么在「只看到局部」的情况下算出「全局正确」的 softmax？答案是 online softmax。

---

## 二、核心原理：tiling + online softmax + recomputation

### 2.1 online softmax 的增量更新

先看 softmax 本身。为了数值稳定，softmax 总是要减去最大值：

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

现在假设一行的 score 被切成两块 $x^{(1)}, x^{(2)}$，我们想分别处理、再合并。定义每块的局部最大值和局部指数和：

$$
m^{(1)} = \max(x^{(1)}), \quad \ell^{(1)} = \sum e^{x^{(1)} - m^{(1)}}
$$

当第二块到来时，更新全局最大值：

$$
m^{\text{new}} = \max(m^{(1)}, m^{(2)})
$$

关键一步：之前用旧的 $m^{(1)}$ 算出的指数和 $\ell^{(1)}$，需要按新旧最大值的差做一次**校正缩放**，再把第二块累加进来：

$$
\ell^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \, \ell^{(1)} + e^{m^{(2)} - m^{\text{new}}} \, \ell^{(2)}
$$

这就是 online softmax 的精髓：用一个 running max $m$ 和 running sum $\ell$，就能在只看局部的情况下，增量地维护出全局正确的归一化分母。

### 2.2 把 V 也一起增量更新

attention 不只要 softmax，还要乘上 $V$。FlashAttention 把输出的累加器 $O$ 也一起做增量更新。遍历 $K, V$ 的每一个 block $j$ 时：

$$
\begin{aligned}
S_{ij} &= Q_i K_j^\top / \sqrt{d} \\
m^{\text{new}} &= \max(m, \ \text{rowmax}(S_{ij})) \\
P_{ij} &= \exp(S_{ij} - m^{\text{new}}) \\
\ell^{\text{new}} &= e^{m - m^{\text{new}}} \ell + \text{rowsum}(P_{ij}) \\
O &= e^{m - m^{\text{new}}} \, O + P_{ij} V_j \\
m, \ell &\leftarrow m^{\text{new}}, \ell^{\text{new}}
\end{aligned}
$$

遍历完所有 block 后，最后做一次归一化 $O \leftarrow O / \ell$。注意整个过程里：

- $S_{ij}$ 和 $P_{ij}$ 只是当前 block 的小矩阵 $[B_q, B_k]$，算完即弃，**从不 materialize 完整的 $[S, S]$**；
- 常驻的只有累加器 $O$（$[B_q, d]$）和两个标量向量 $m, \ell$（$[B_q]$）。

写成伪代码就是经典的双重循环（FA1 的视角，外层 K/V、内层 Q）：

```python
# Q, K, V: [S, d]；分块大小 Bq, Bk
for j in range(0, S, Bk):                 # 外层遍历 K/V block
    Kj, Vj = K[j:j+Bk], V[j:j+Bk]
    for i in range(0, S, Bq):             # 内层遍历 Q block
        Qi = Q[i:i+Bq]
        Sij = (Qi @ Kj.T) / sqrt(d)       # [Bq, Bk]，只在 SRAM
        m_new = maximum(m[i], rowmax(Sij))
        Pij = exp(Sij - m_new)
        l_new = exp(m[i] - m_new) * l[i] + rowsum(Pij)
        O[i]  = exp(m[i] - m_new) * O[i] + Pij @ Vj
        m[i], l[i] = m_new, l_new
O = O / l[:, None]
```

### 2.3 IO 复杂度：为什么这样就快了

设 on-chip SRAM 大小为 $M$。标准 attention 的 HBM 访问量是 $O(S^2)$（要写读完整的 score 矩阵）。FlashAttention 通过分块，把 HBM 访问量降到：

$$
O\left(\frac{S^2 d^2}{M}\right)
$$

由于 $d^2$ 通常远小于 $M$（例如 $d=128$，$d^2 = 16384$，而 SRAM 可容纳的元素数更大），这个量比 $O(S^2)$ 小很多。换句话说，**FlashAttention 没有减少计算量（FLOPs 还是 $O(S^2 d)$），但大幅减少了 HBM 读写量**，把瓶颈从内存搬运转回了实际计算。

### 2.4 recomputation：反向不存 attention 矩阵

反向传播需要 attention 概率 $P$ 来求梯度。标准实现会把 forward 的 $[S,S]$ 存下来给 backward 用——但这正是我们想避免的。FlashAttention 的做法是：**只存下 $O$ 和 logsumexp 统计量 $L = m + \log \ell$（每行一个标量），backward 时用它们重新算出 $P$**。

这是一种典型的「用计算换显存」：多花一点重算的 FLOPs，换掉 $O(S^2)$ 的 activation 存储。由于 attention 本身的算力在重算后仍远快于额外的 HBM 往返，这笔交易在长序列下非常划算。

---

## 三、FlashAttention-1：把算法落成 kernel

FlashAttention-1（2022）的贡献是把上面这套 IO-aware 算法第一次落成了高效的 CUDA kernel，并证明了它在端到端训练里的价值：

- **显存**：attention 部分从 $O(S^2)$ 降到 $O(S)$，使得训练能支持远更长的上下文；
- **速度**：在 GPT-2/GPT-3 规模上，相比当时 PyTorch 的标准实现有数倍加速；
- **精确**：它是**精确 attention**，不是近似——输出和标准 attention 在数值上等价（仅有浮点误差）。

FA1 把一个 kernel 的工作划分成「外层循环遍历 K/V block、内层循环遍历 Q block」的结构。这个划分在当时是自然的，但也埋下了 FA2 要解决的低效问题。

---

## 四、FlashAttention-2：更好的并行与工作划分

FA1 虽然已经把 attention 变成 IO-aware 的分块计算，但它在 GPU 上的实际吞吐只有理论峰值的 25–40%——离优化过的 GEMM 还差得远。FlashAttention-2（2023）的目标不是改算法的数学，而是改它在 GPU 上的**工作划分与并行方式**。三个关键改动：

### 4.1 减少 non-matmul FLOPs

现代 GPU 上 matmul 和非 matmul 运算的算力差距巨大。以 A100 为例，FP16/BF16 matmul 的峰值是 312 TFLOPs/s，而非 matmul 的 FP32 运算只有 19.5 TFLOPs/s——**每个 non-matmul FLOP 大约比 matmul FLOP 贵 16 倍**。online softmax 里的那些 `exp`、rescale、rowmax/rowsum 都属于 non-matmul，虽然占总 FLOPs 比例不高，却会拖慢整体。

FA2 重写了 online softmax，尽量减少校正缩放的次数。核心技巧是：**在内层循环中不再每步都对 $O$ 做除以 $\ell$ 的归一化，而是把 rescale 推迟到最后**，中间只维护未归一化的累加器，每行只保存一个 logsumexp 标量：

$$
\tilde{O} = \sum_j e^{S_{ij} - m^{\text{final}}} V_j, \qquad O = \tilde{O} / \ell^{\text{final}}
$$

这样把原本每个 block 都要做的 rescale，压缩成整行结束后的一次，省下大量 non-matmul 运算。

### 4.2 沿 sequence length 维并行

FA1 只在 batch × head 维度上做线程块（thread block）并行。当 batch 小、序列长时（正是长上下文训练的典型场景），线程块数量不足，GPU 的 SM 占用率（occupancy）很低，大量计算单元闲着。

FA2 让单个 head 内部也能沿 **sequence length 维**拆到不同线程块上并行。这样即使 batch=1，长序列也能产生足够多的线程块把 GPU 喂满。

### 4.3 split-Q 取代 split-K

这是 FA2 最巧妙的一处。在一个线程块内部，工作还要再分给若干 warp。FA1 采用 **split-K**：每个 warp 负责 K/V 的一部分，但因为 softmax 要跨整行归一化，各 warp 算完后必须通过 shared memory 交换中间结果、再汇总，产生大量 warp 间通信。

FA2 改成 **split-Q**：每个 warp 负责一部分 Q 行，独立地遍历完整的 K/V。由于每个 warp 处理的是不同的 query 行、互不依赖，**warp 之间几乎不需要通过 shared memory 通信**。

```text
split-K（FA1）：warp 切 K/V → 各 warp 持有部分行的部分和 → 必须跨 warp 归约
split-Q（FA2）：warp 切 Q   → 各 warp 独立算完整行            → 几乎零 warp 间通信
```

### 4.4 结果

三者叠加，FA2 相比 FA1 约 **2× 加速**，达到理论峰值的 50–73%；在 A100 上 FP16/BF16 可达 ~230 TFLOPs/s，端到端训练的 MFU（model FLOPs utilization）可到 72%。实现上 FA2 也基于 NVIDIA 的 CUTLASS 3.x / CuTe 从头重写。

---

## 五、FlashAttention-3：拥抱 Hopper 的异步

FA2 虽然在 A100（Ampere）上很高效，但搬到 H100（Hopper）上时利用率只有约 35%。原因是 Hopper 引入了一批新的异步硬件能力，而 FA2 的同步执行模型没有用上它们。FlashAttention-3（2024）专门针对 Hopper 重写，核心是**让数据搬运和计算异步重叠起来**。三大技术：

### 5.1 warp-specialization：producer-consumer 异步

FA3 把一个线程块里的 warp group 分成两类角色，组成一条流水线：

- **producer warp**：只负责用 **TMA**（Tensor Memory Accelerator，Hopper 上专门搬数据的硬件单元）把下一个 K/V block 从 HBM 预取到 shared memory；
- **consumer warp**：专注算 matmul（**WGMMA**，Hopper 新一代 Tensor Core 指令，吞吐远高于 Ampere 的 `mma.sync`）和 softmax。

这样当 consumer 在算当前 block 时，producer 已经在搬下一个 block 的数据，**计算和访存彻底重叠**，不再互相等待。一个关键使能特性是 Hopper 支持用 `setmaxnreg` 在 warp group 之间**动态分配寄存器**：负责 WGMMA 的 consumer 拿到更多寄存器（用来开更大的 tile），而只需一个线程发 TMA 指令的 producer 占用极少寄存器。

### 5.2 matmul 与 softmax 的交织 overlap

即使在 consumer 内部，matmul（走 Tensor Core）和 softmax（走普通 CUDA core 的 `exp` 等）也是两类不同的硬件单元。FA3 把相邻 block 的这两类操作**交织**起来：在对当前 block 做 softmax 的同时，让 Tensor Core 去算下一个 block 的 $QK^\top$。这样 Tensor Core 不会因为等 softmax 而空转，进一步压满算力。

### 5.3 FP8 低精度 + block quantization + incoherent processing

Hopper 的 Tensor Core 在 FP8 下吞吐翻倍（FP16 ~989 TFLOPs/s vs FP8 ~1978 TFLOPs/s）。但 FP8 位数少，直接用会有明显精度损失。FA3 用两个技巧把误差压回来：

- **block quantization**：不对整个张量用一个 scale，而是按 block 分别量化，让每块用更贴合自己数值范围的 scale；
- **incoherent processing**：在量化前用随机正交矩阵（如 Hadamard 变换）把 $Q, K$ 旋转一下，把个别异常大的 outlier「摊平」到各维度，从而减小量化误差。

### 5.4 结果

FA3 在 H100 上：BF16 达到 ~840 TFLOPs/s（**85% 利用率**），相比 FA2 有 1.5–2.0× 加速；FP8 接近 1.3 PFLOPs/s，且数值误差比基线 FP8 attention 小 **2.6×**。

### 5.5 三代演进一张表

| | FA1 (2022) | FA2 (2023) | FA3 (2024) |
|---|---|---|---|
| 关键词 | IO-aware、tiling | 工作划分、并行 | 异步、低精度 |
| 主要瓶颈 | HBM 读写 $O(S^2)$ | GPU occupancy 低、warp 通信多 | Hopper 异步能力没用上 |
| 核心改动 | online softmax + recompute | 减 non-matmul FLOPs、seq 并行、split-Q | warp-specialization、TMA/WGMMA、FP8 |
| 目标硬件 | 通用（A100 为主） | Ampere（A100） | Hopper（H100） |
| 利用率 | 25–40% | 50–73% | 75–85%（BF16） |

可以看到一条清晰的主线：**FA1 解决「要不要把 S×S 写进 HBM」，FA2 解决「怎么把工作均匀铺满 GPU」，FA3 解决「怎么让搬运和计算异步起来、并用上低精度」**。算法的数学骨架（tiling + online softmax + recompute）三代未变，变的是它如何贴合不断演进的 GPU 硬件。

---

## 六、不同模型结构上的定制化开发

实际项目里，你的模型往往不是「标准 full attention」：可能是 causal、sliding window、加了 ALiBi/相对位置偏置、用了 MQA/GQA、需要变长序列打包，甚至有一个完全自定义的 mask。这时该怎么用上 FlashAttention？分两个层面看。

### 6.1 接口 / 集成层：能配置的就别改 kernel

绝大多数常见变体，官方 `flash-attn` 已经内置参数支持，根本不用碰 CUDA。这一层应该是你的首选。

**causal mask**：最常见的需求，一个参数搞定。FA kernel 内部会跳过被 mask 的整个 block（上三角部分根本不计算），所以 causal 不仅正确还更快：

```python
from flash_attn import flash_attn_func

# q, k, v: [batch, seqlen, nheads, head_dim]
out = flash_attn_func(q, k, v, causal=True)
```

**sliding window（局部注意力）**：Mistral 等模型用的滑动窗口，FA 直接支持 `window_size` 参数，只计算窗口内的 block：

```python
out = flash_attn_func(q, k, v, causal=True, window_size=(4096, 0))  # 左 4096，右 0
```

**MQA / GQA**：多个 query head 共享一组 KV head。不需要把 KV 物理复制成和 Q 一样多（那会浪费显存和带宽），直接传入 head 数更少的 K/V，FA 内部处理这种 broadcast：

```python
# q: [b, s, 32, d]，k/v: [b, s, 4, d]  → 8 个 query head 共享 1 个 kv head
out = flash_attn_func(q, k, v, causal=True)   # GQA：nheads_q=32, nheads_kv=4
```

**变长序列打包（varlen）**：训练时为了不浪费算力，常把不同长度的样本拼成一个无 padding 的长序列，用 `cu_seqlens`（累积长度前缀）标记边界。FA 的 varlen 接口能在拼接序列上做到「样本之间互不注意」：

```python
from flash_attn import flash_attn_varlen_func

# q,k,v: [total_tokens, nheads, d]；cu_seqlens: [batch+1]，如 [0, 12, 30, 30+...]
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_len, max_seqlen_k=max_len,
    causal=True,
)
```

**ALiBi 等位置偏置**：FA 支持传入 `alibi_slopes`，在 kernel 内部把线性偏置加到 score 上，而不必在外面 materialize 一个 $[S,S]$ 的 bias 矩阵（那又会把我们省下的显存吃回去）。

**经验法则**：如果你的需求是 causal、window、GQA/MQA、varlen、ALiBi 这类「主流变体」，几乎一定有现成参数——先翻文档，别急着写 kernel。

### 6.2 kernel 改写层：当配置参数不够用时

如果你的 attention 有一个**任意的、数据相关的 score 修改**（比如一个无法用现成参数表达的自定义 mask，或者一种新的相对位置偏置、新的 attention 变体），就需要改 kernel 了。这时不建议直接动官方的 CUDA/CUTLASS 代码——门槛极高——而是用 **Triton** 写一个 FlashAttention 风格的 kernel，在分块循环里插入你的逻辑。

关键是理解：所有自定义都发生在「算完 $S_{ij}$、做 online softmax 之前」这个钩子点上。拿到当前 block 的 score 子矩阵后，加 bias、应用 mask、做任何 element-wise 变换，再交给后续的 max/exp/累加：

```python
# Triton FlashAttention 内层循环的示意（省略 boilerplate）
qk = tl.dot(q, k) * sm_scale                 # [Bq, Bk] 当前 block 的 score

# ---- 自定义钩子点：在这里改 score ----
qk = qk + my_custom_bias(offs_q, offs_k)     # 例如自定义相对位置偏置
qk = tl.where(my_custom_mask(offs_q, offs_k), qk, float("-inf"))  # 自定义 mask
# -------------------------------------

m_new = tl.maximum(m_i, tl.max(qk, axis=1))  # online softmax，照常
p = tl.exp(qk - m_new[:, None])
l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)
acc = acc * tl.exp(m_i - m_new)[:, None] + tl.dot(p.to(v.dtype), v)
m_i, l_i = m_new, l_new
```

几个落地要点：

- **mask 要在 block 级别短路**：如果某个 $(Q_{\text{block}}, K_{\text{block}})$ 整块都被 mask 掉（比如 causal 的上三角块），直接 `continue` 跳过，别白算——这正是 causal 比 full 快的原因，自定义 mask 也应沿用。
- **backward 要自己配套**：改了 forward 的 score 逻辑，反向的梯度公式也得相应推导。Triton 里通常要手写对应的 backward kernel，或确保你的 bias/mask 对输入的梯度被正确处理（很多自定义 bias 本身带可学习参数，梯度不能漏）。
- **数值稳定性别破坏**：自定义变换要兼容 online softmax 的减最大值机制。比如加 bias 后再求 max，不要在减 max 之前就把 bias 丢掉。
- **优先找现成的可编程框架**：PyTorch 的 `FlexAttention` 提供了 `score_mod` 和 `mask_mod` 两个钩子，让你用纯 Python 函数表达「如何改 score / 如何 mask」，再由编译器生成融合 kernel。它正是为「我想要一个新变体但不想手写 Triton」这个场景设计的，应优先考虑。

### 6.3 选择哪一层：一个决策顺序

1. 需求是主流变体（causal/window/GQA/varlen/ALiBi）→ **用 `flash-attn` 参数**；
2. 需求是自定义 score/mask，但能表达成 element-wise 函数 → **用 FlexAttention 的 `score_mod`/`mask_mod`**；
3. 需求非常特殊、或要极致性能、或要上新硬件 → **用 Triton 手写 FA 风格 kernel**，在 online softmax 前的钩子点插入逻辑，并配套 backward。

层级越往下，灵活性越高，但开发和维护成本也越高。绝大多数模型工程师停在第 1、2 层就够了。

---

## 七、总结

FlashAttention 不是一个「近似」或「省显存的小技巧」，而是对 attention 计算方式的一次重写：它用 **tiling + online softmax + recomputation**，在数学上等价地避免了 materialize $O(S^2)$ 的注意力矩阵，把一个 memory-bound 的操作变回了 compute-bound。

三代的演进是「同一套算法不断贴合硬件」的过程：

- **FA1** 确立 IO-aware 思想，不把 $S \times S$ 写进 HBM；
- **FA2** 通过减少 non-matmul FLOPs、沿序列维并行、split-Q，把 GPU 喂满；
- **FA3** 用 warp-specialization、TMA/WGMMA 异步和 FP8，吃透 Hopper 的硬件能力。

而在自己的模型上做定制时，正确的顺序是**从接口层往 kernel 层走**：能用参数配置的（causal、window、GQA、varlen、ALiBi）就别写 kernel；需要自定义 score/mask 的优先用 FlexAttention 这类可编程框架；只有在极特殊或追求极致性能时，才下沉到 Triton 手写 kernel，并记住所有自定义都应发生在「online softmax 之前的那个钩子点」上，且 forward 与 backward 必须配套。

理解了这套结构，FlashAttention 对你就不再是一个黑盒开关，而是一个可以按需裁剪、扩展、甚至移植的计算原语。

---

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 — PyTorch Blog](https://pytorch.org/blog/flashattention-3/)
- [FlashAttention-3 — Colfax Research 技术详解](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/)



