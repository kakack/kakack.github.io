---

layout: post
tags: [LLM, NLP, AI Infra]
title: 并行策略到底在切什么：训练 OOM 与 ZeRO/DP/TP/SP 的边界
date: 2026-6-30
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

在训练大模型时，OOM 并不总是同一种问题。最常见的误解是：只要把 `ZeRO-3` 开起来、把 DP 卡数加上去，或者多加几台机器，显存问题就应该线性缓解。但实际训练里经常会遇到另一类 OOM：某个 step 内部临时产生了一个巨大的局部张量，例如长序列下的 attention scores、activation，或者更隐蔽的 `logits` / cross entropy backward buffer。它们不是模型状态，也不是 optimizer state，所以不会因为 ZeRO 或数据并行自动被切开。

换句话说，**单卡显存峰值由「常驻的模型状态」和「瞬时的计算张量」两部分共同决定**，而各种并行策略往往只对其中一部分起作用。把它们混为一谈，是长序列训练里最常踩的坑。

我最近遇到的现象就是：当 sequence length 非常长时，训练过程中的 `logits` 以及 loss backward 产生的中间张量会被一次性塞进 GPU 显存。其中最大的那个张量尺寸通常接近：

$$
\text{logits} \in \mathbb{R}^{B \times S \times V}
$$

其中 $B$ 是 micro batch size，$S$ 是 sequence length，$V$ 是 vocabulary size。它不是参数，也不是梯度或优化器状态，而是当前 rank 在 forward / backward 中产生的激活类临时张量。因此，如果只是增加数据并行卡数、增加节点数，或者使用 ZeRO-3 分片参数，它仍然可能完整地出现在每张参与训练的 GPU 上。

这篇文章想讨论的不是“怎么无脑省显存”，而是一个更实用的问题：**训练 OOM 到底发生在哪类内存上？它能不能被 ZeRO、DP、TP、SP 或加节点解决？**

---

## 一、训练显存先分成几类

LLM 训练中的 GPU 显存大致可以拆成几类：

1. **模型参数（parameters）**
2. **参数梯度（gradients）**
3. **优化器状态（optimizer states）**
4. **前向激活（activations）**
5. **临时计算 buffer / workspace**
6. **输出 logits 与 loss backward 相关中间量**

不同并行策略解决的对象不同。很多时候，OOM 的根源不是“模型太大”，而是“某个局部张量太大”。如果没有先判断 OOM 属于哪一类，很容易把 ZeRO、DP、TP、SP 混在一起，以为它们都能等价地降低单卡显存。

### 1.1 参数、梯度、优化器状态

以 AdamW 训练为例，除了参数本身，还需要梯度、一阶动量、二阶动量。如果使用混合精度训练，通常还会保留 FP32 master weights。粗略看，一个参数可能对应多份状态：

- FP16 / BF16 parameter
- FP16 / BF16 gradient
- FP32 master parameter
- FP32 momentum
- FP32 variance

这部分显存是 ZeRO / FSDP 最擅长处理的对象。尤其是 ZeRO-3 会把参数、梯度和 optimizer states 都在数据并行 rank 之间切分，能显著降低单卡常驻显存。

### 1.2 Activations

Activation 是 forward 过程中为了 backward 保存下来的中间结果。它们的规模通常与以下因素相关：

$$
O(B \times S \times H \times L)
$$

其中 $H$ 是 hidden size，$L$ 是层数。对于长序列训练，activation 往往比模型状态更早成为瓶颈。Gradient checkpointing 通过在 backward 时重算部分 forward，减少保存的 activation，是缓解这类 OOM 的常见手段。

但 activation 是否能被并行切开，要看它落在哪个维度上。Tensor Parallelism 通常切 hidden / head / FFN 中间维度；Sequence Parallelism 则尝试切 sequence 维度上的 activation；Context Parallelism 更进一步面向超长上下文，把 attention 的上下文维度跨设备切分。

### 1.3 临时张量与 workspace

还有一类显存来自算子内部临时 buffer，例如 GEMM workspace、attention score、softmax buffer、mask buffer、通信 buffer 等。它们通常不长期存在，但峰值可能很高。OOM 往往发生在这些临时张量和 activation、logits 同时存在的瞬间。

FlashAttention 这类 kernel 的价值就在于避免显式 materialize 巨大的 $S \times S$ attention matrix，将 attention 从显存密集型操作改造成更接近 IO-aware 的分块计算。

### 1.4 Logits 与 loss backward

最后一类就是这篇文章的重点：`logits` 和 loss backward 中间量。

语言模型训练通常会经过最后的 `LM Head`：

$$
\text{hidden states}: [B, S, H]
$$

$$
\text{lm head weight}: [H, V]
$$

$$
\text{logits}: [B, S, V]
$$

当 $S$ 很长、$V$ 很大时，`logits` 本身就可能非常夸张。假设：

- micro batch size $B=1$
- sequence length $S=128K$
- vocab size $V=128K$
- dtype 为 BF16，即 2 bytes

那么仅 logits 就接近：

$$
1 \times 128K \times 128K \times 2 \approx 32\text{GB}
$$

而这还只是 forward 产生的 logits 张量本身。真正致命的往往是 backward：cross entropy 求梯度时通常需要先做一次 `softmax`，得到一个与 logits 同形状的概率张量 $[B,S,V]$，再产生同样大小的 `grad_logits`。如果框架还以 FP32 计算 softmax（数值稳定性的常见做法），单这一步就可能再吃掉 $1 \times 128K \times 128K \times 4 \approx 64\text{GB}$。也就是说，**logits 的显存峰值常常是「logits 本体」的 2~3 倍**，远不止那 32GB。

这种 OOM 的直觉很反常：模型参数可能已经被 ZeRO-3 切得很好，attention 也可能使用了 FlashAttention，但最后一个 vocabulary projection 和 loss 仍然能把单卡显存打爆。它出现在计算图的最末端，体积却可能超过前面任意一层。

问题的根源就藏在最朴素的写法里：

```python
# hidden_states: [B, S, H]
logits = self.lm_head(hidden_states)          # [B, S, V]，一次性 materialize
loss = F.cross_entropy(
    logits.view(-1, V).float(),               # 再 cast 一份 FP32，峰值翻倍
    labels.view(-1),
)
```

`lm_head` 这一行就实打实地分配了 $[B,S,V]$，`.float()` 又复制了一份，backward 还要为它保留 autograd 所需的中间量。这几行代码本身没有任何 bug，但在长序列、大词表下，它们就是单卡 OOM 的直接来源。后文讨论的所有手段，本质上都是在改写这一小段逻辑。

---

## 二、DP 和 ZeRO 能解决什么，不能解决什么

### 2.1 Data Parallelism

Data Parallelism（DP）的核心是每张卡持有一份完整模型，处理不同的数据分片，backward 后做梯度同步。

它能改善的是吞吐，而不是天然降低单卡显存。对于同样的 per-GPU micro batch size，增加 DP 卡数不会降低每张卡上的：

- 参数
- activation
- logits
- loss backward buffer

只有当“全局 batch size 固定”且你把 batch 拆到更多 GPU 上，使每张卡的 micro batch size 变小时，DP 才会间接降低 activation 和 logits 的 per-rank 尺寸。但这不是 DP 本身切分了张量，而是你改变了每张卡实际处理的数据量。

所以：

- **如果 OOM 来自全局 batch 太大**：增加 DP 卡数并降低 per-GPU batch，可能有效；
- **如果 OOM 来自单样本超长序列**：DP 基本无效，因为 $B$ 已经不能再降，$S$ 仍然完整地在每张卡上。

### 2.2 ZeRO

ZeRO 的不同 stage 主要处理模型状态：

- **ZeRO-1**：切 optimizer states；
- **ZeRO-2**：切 optimizer states + gradients；
- **ZeRO-3**：切 optimizer states + gradients + parameters。

因此 ZeRO 对“模型状态 OOM”非常有效。例如模型参数、梯度和 Adam 状态太大，ZeRO-3 / FSDP 能明显降低单卡显存。

但 ZeRO 对以下对象通常无能为力：

- 当前 rank 的 activation；
- 当前 rank 的 attention 临时张量；
- 当前 rank 的 logits；
- 当前 rank 的 cross entropy backward 中间量；
- 算子 workspace。

原因很简单：ZeRO 切的是模型状态，不是任意 forward 中间结果。ZeRO-3 可以让参数在计算前 all-gather、计算后释放，但如果某个算子本身需要产生 $[B,S,V]$ 的 logits，那么这个局部输出仍然会出现在执行该算子的设备上。

---

## 三、TP 能解决什么，不能解决什么

Tensor Parallelism（TP）把一个层内部的矩阵乘法切到多张 GPU 上。典型做法包括：

- column parallel linear；
- row parallel linear；
- attention heads 切分；
- FFN intermediate dimension 切分；
- vocab parallel embedding / LM head。

TP 对大模型训练很关键，因为它可以降低单卡参数、梯度、optimizer state 以及部分 activation / intermediate 的尺寸。对于本文讨论的 logits OOM，TP 是否有效取决于最后的 LM Head 和 loss 是否也做了 **vocab parallel**。

### 3.1 普通 LM Head：可能仍然 OOM

如果每张卡最终都 all-gather 到完整 hidden states，并在本地得到完整 logits：

$$
[B,S,H] \times [H,V] \rightarrow [B,S,V]
$$

那么 TP 对 logits 峰值帮助有限。即使中间层被 tensor parallel 切开，最后完整的 $[B,S,V]$ 仍然可能落到某张卡或每张卡上。

### 3.2 Vocab Parallel LM Head：可以缓解 logits OOM

更合理的方式是把 vocab 维度切开：

$$
\text{logits}_i \in \mathbb{R}^{B \times S \times \frac{V}{TP}}
$$

每个 TP rank 只负责一部分 vocabulary。配合 vocab-parallel cross entropy，可以在不 materialize 全量 logits 的情况下完成 loss 计算。Megatron-LM 系列训练框架中的 `VocabParallelCrossEntropy` 就是这一类思想。

这种情况下，TP 能直接降低 logits 的单卡显存：

$$
O(B \times S \times V) \rightarrow O(B \times S \times \frac{V}{TP})
$$

但要注意：只切 LM Head 不够，loss 也必须识别这种 vocab-parallel logits。如果先 all-gather 出完整 logits 再算 cross entropy，显存峰值又回来了。

### 3.3 TP 的边界

TP 并不是万能的。它通常不能解决：

- 单个样本 sequence length 太长导致的 $S$ 维 activation 爆炸；
- 未分块 cross entropy 造成的 $B \times S \times V$ 峰值；
- attention 中显式 materialize 的 $S \times S$ scores；
- 非 TP-aware 算子内部产生的完整中间张量。

TP 只在计算图和算子实现都真正按 tensor 维度切分时才有效。否则它可能只是切了中间层参数，最后仍然在某些边界处 all-gather 回完整张量。

---

## 四、SP / CP 能解决什么，不能解决什么

Sequence Parallelism（SP）和 Context Parallelism（CP）经常被放在一起讨论，但它们解决的问题略有不同。

### 4.1 Sequence Parallelism

SP 的目标是把 sequence 维度上的 activation 分散到不同 GPU 上，典型场景是和 TP 配合，减少 transformer layer 中不能被 TP 自然切分的 activation。例如 LayerNorm、Dropout、Residual 相关的中间状态，如果每张 TP rank 都保留完整 $[B,S,H]$，会造成显存浪费。SP 可以把它变成：

$$
[B,S,H] \rightarrow [B,\frac{S}{SP},H]
$$

因此 SP 对长序列 activation OOM 是有效的，尤其是在 sequence length 很长时。

> 需要提醒的是，「Sequence Parallelism」在不同框架里指代并不完全相同。Megatron-LM 语境下的 SP 通常**和 TP 绑定**，专门用来切分 TP 无法自然分摊的那部分 activation（LayerNorm、Dropout、residual）；而 DeepSpeed-Ulysses 语境下的「序列并行」更接近本文 CP 一节讨论的、面向超长上下文的 attention 切分。看文档时先确认它到底切的是哪一段计算，再判断它能不能解决你的 OOM。

但 SP 是否能解决 logits OOM，取决于最后的 LM Head 和 loss 是否也沿 sequence 维度分片。如果训练框架在 loss 前把 sequence gather 回完整 $S$，那么 logits 仍然会恢复成 $[B,S,V]$。如果能够保持 sequence-sharded hidden states，并分片计算 loss，则 logits 可以降为：

$$
[B,\frac{S}{SP},V]
$$

进一步如果同时使用 vocab parallel，则变成：

$$
[B,\frac{S}{SP},\frac{V}{TP}]
$$

这才是长序列 + 大词表训练中更理想的形态。

### 4.2 Context Parallelism

CP 更关注 attention 的上下文长度问题。标准 attention 的 score 规模是：

$$
[B, heads, S, S]
$$

即使使用 FlashAttention 不显式保存完整矩阵，KV、activation、通信和 backward 仍然会随着上下文变长变得昂贵。CP / ring attention / Ulysses / context parallel 这类方法尝试把上下文维度跨设备切分，让每张卡只处理一部分 sequence context，再通过通信完成全局 attention。

所以 CP 更适合解决：

- 超长上下文 attention 计算；
- KV / activation 随 $S$ 增长造成的单卡压力；
- 单样本长序列无法靠 DP 拆分的问题。

但 CP 本身不一定解决 logits OOM。因为 logits 的大头是 $S \times V$，不是 attention 的 $S \times S$。如果 loss 侧没有 sequence-sharded 或 chunked 实现，CP 之后仍然可能在输出层重新聚合成长序列 logits。

---

## 五、增加节点什么时候有用

“加机器”本身不是一种显存优化，它只是提供了更多 GPU。是否降低单卡显存，取决于新增 GPU 被用于哪种并行维度。

### 5.1 有用的情况

增加节点有用，通常是因为你把新增 GPU 用到了模型并行或序列并行上：

- 增大 ZeRO / FSDP 的 shard group，进一步切分模型状态；
- 增大 TP，切分层内矩阵、attention head、vocab；
- 增大 SP / CP，把 sequence 或 context 分散到更多 GPU；
- 增大 PP，把层数分摊到不同 stage；
- 在全局 batch 固定时增大 DP，从而降低 per-GPU micro batch。

### 5.2 没用的情况

如果新增节点只增加 DP world size，但每张卡处理的 micro batch、sequence length、vocab logits 形状不变，那么单卡峰值通常不变。

典型无效场景：

- $B=1$，已经无法继续降低 micro batch；
- 单条样本 sequence length 极长；
- logits 在每个 rank 上都是完整 $[B,S,V]$；
- loss backward 需要完整 logits/probability buffer；
- attention kernel 或 loss kernel 不是分布式实现。

这也是为什么“更多节点”不一定解决长序列 OOM。它只提供了拆分的可能性，不会自动拆分所有张量。

---

## 六、按 OOM 类型看解决方式

下面是一个更实用的排查表：

| OOM 来源 | 典型张量 | ZeRO / FSDP | DP / 加节点 | TP | SP / CP | 更直接的手段 |
|---|---|---:|---:|---:|---:|---|
| 参数太大 | weights | 有效 | 仅 DP 无效 | 有效 | 间接 | ZeRO-3、FSDP、TP、PP |
| optimizer state 太大 | Adam states | 非常有效 | 仅 DP 无效 | 有效 | 无直接帮助 | ZeRO-1/2/3、8-bit optimizer |
| 梯度太大 | gradients | 有效 | 仅 DP 无效 | 有效 | 间接 | ZeRO-2/3、FSDP |
| activation 太大 | $B,S,H,L$ | 部分无效 | 降低 per-GPU batch 才有效 | 部分有效 | 有效 | checkpointing、SP、CP、降低 micro batch |
| attention scores 太大 | $B,heads,S,S$ | 无效 | 通常无效 | 部分有效 | CP 有效 | FlashAttention、ring attention、CP |
| logits 太大 | $B,S,V$ | 无效 | 通常无效 | vocab parallel 有效 | sequence-sharded loss 有效 | vocab-parallel CE、chunked CE |
| loss backward buffer 太大 | probs / grad logits | 无效 | 通常无效 | 取决于 loss 实现 | 取决于 loss 实现 | fused CE、chunked CE、recompute logits |
| 通信 buffer / workspace | temp buffer | 通常无效 | 通常无效 | 不确定 | 不确定 | 调 kernel、bucket size、算子实现 |

这里最容易被误判的是 logits 和 loss。因为它们出现在模型最后，看起来只是一个普通输出，但尺寸是 $B \times S \times V$。当 $S$ 和 $V$ 同时很大时，它甚至可能比任意一层 activation 更大。

---

## 七、长序列 logits OOM 怎么处理

回到开头的问题：长 sequence length 下，`logits` 和 loss backward 被一次性塞进 GPU，ZeRO-3、DP、增加节点都不能改善，应该怎么办？

### 7.1 确认是不是完整 logits

先确认训练框架是否真的 materialize 了完整 logits：

$$
[B,S,V]
$$

可以从几个角度排查：

- OOM 是否发生在 `lm_head`、`cross_entropy`、`loss.backward` 附近；
- 显存峰值是否随 vocab size 线性增长；
- 显存峰值是否随 sequence length 线性增长；
- 开启 ZeRO-3 后模型状态下降，但峰值仍然几乎不变；
- 减小 micro batch 或 sequence length 后立即缓解。

如果这些现象同时出现，基本可以判断不是 ZeRO 管辖范围内的问题。

比起靠现象猜，更可靠的是直接让 PyTorch 把峰值显存的来源打印出来。两个常用工具：

```python
# 1) 看一个 step 内的峰值，定位 OOM 发生在哪一段
torch.cuda.reset_peak_memory_stats()
logits = model.lm_head(hidden_states)
loss = loss_fn(logits, labels)
print(f"after forward: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
loss.backward()
print(f"after backward: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# 2) 抓一段完整的分配轨迹，用官方可视化工具逐张量回溯
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... 跑一两个 step ...
torch.cuda.memory._dump_snapshot("mem.pickle")   # 上传到 https://pytorch.org/memory_viz 查看
```

如果峰值正好在 `lm_head` 之后、`backward` 之前那一段陡增，并且增量约等于 $B \times S \times V \times \text{dtype}$，那就基本坐实是 logits 而非模型状态的问题。

### 7.2 使用 vocab-parallel cross entropy

如果已经使用 TP，优先确认最后的 LM Head 和 cross entropy 是否是 vocab parallel 的。目标是让每张 TP rank 只持有：

$$
[B,S,V/TP]
$$

并且 loss 在分布式 logits 上直接计算，而不是 all-gather 完整 logits 后再算。

这类实现通常需要处理：

- 每个 rank 只覆盖一段 vocab range；
- label 所在 vocab range 之外的 rank 参与 global max / sum-exp；
- softmax normalization 跨 rank reduce；
- 最终 loss 只取 label 对应 rank 的 logit；
- backward 时生成局部 grad logits。

如果这些逻辑没有在 loss 内部完成，TP 对 logits 显存的帮助会被 all-gather 抵消。

### 7.3 使用 sequence-sharded loss

如果使用 SP / CP，理想情况下不要在 loss 前 gather 回完整 sequence。让每张 rank 只处理一段 token：

$$
[B,S/SP,H] \rightarrow [B,S/SP,V]
$$

然后对 token-level loss 做 reduce。这样可以把 logits 沿 $S$ 维切开。

进一步结合 vocab parallel：

$$
[B,S,V] \rightarrow [B,S/SP,V/TP]
$$

这对长序列尤其关键，因为 $S$ 是 OOM 的主要放大器。

### 7.4 Chunked cross entropy

如果框架暂时不支持分布式 loss，可以考虑 chunked cross entropy：不要一次性计算全部 token 的 logits，而是按 sequence chunk 分块：

$$
[B,S,H] \rightarrow [B,C,H] \rightarrow [B,C,V]
$$

每次只计算长度为 $C$ 的 logits 和 loss，累积 token loss。为了 backward 正确，通常有两类实现：

- 保存较少中间量，在 backward 中重算 logits；
- 使用 fused / custom autograd function，把 forward 和 backward 的 buffer 控制在 chunk 内。

这会增加计算量或实现复杂度，但能把 logits 峰值从 $O(BSV)$ 降到 $O(BCV)$。

最朴素的版本甚至不需要自定义 autograd，只要把 sequence 切块、让每个 chunk 各自走一次 `cross_entropy` 并按 token 数加权累加即可。但这里有个**容易写错的关键点**：如果只是把各 chunk 的 loss 累加成一个标量、最后统一 `.backward()`，那么每个 chunk 的 logits 计算图都会被 autograd 一直保留到反向，峰值并不会下降。真正省显存的写法是让每个 chunk 立刻反向、随即释放它的图：

```python
def chunked_cross_entropy_backward(hidden_states, lm_head, labels, chunk_size=2048):
    # hidden_states: [B, S, H]（requires_grad），labels: [B, S]
    B, S, H = hidden_states.shape
    total_loss = 0.0
    total_tokens = labels.ne(-100).sum().clamp(min=1)
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        h_chunk = hidden_states[:, start:end, :]      # [B, C, H]
        logits = lm_head(h_chunk).float()             # [B, C, V]，仅此 chunk 存活
        lbl = labels[:, start:end].reshape(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), lbl,
            ignore_index=-100, reduction="sum",
        ) / total_tokens
        loss.backward()                               # 立即反向，随后该 chunk 的 logits 图被释放
        total_loss += loss.detach()
    return total_loss
```

每个 chunk 在 `loss.backward()` 后就把自己的 $[B,C,V]$ logits 和概率张量释放掉，梯度则累加进 `hidden_states.grad` 和 `lm_head` 的参数梯度里。于是同一时刻只有一个 chunk 的 logits 存活，峰值从 $O(BSV)$ 降到 $O(BCV)$。代价是把一次大的 matmul + loss 拆成了多次小的，kernel launch 和访存效率略有损失。生产环境里更进一步的做法（如 Liger-Kernel 的 fused linear cross entropy、cut cross entropy）会把 projection 和 loss 融进一个 kernel，连 chunk 的完整 logits 都不 materialize，把峰值压到接近 $O(BC)$。

### 7.5 降低 micro batch 或做梯度累积

如果 $B>1$，降低 micro batch size 是最直接的办法：

$$
O(B \times S \times V)
$$

中的 $B$ 线性下降。为了保持全局 batch size，可以增加 gradient accumulation steps。

但如果已经是 $B=1$，这个方向就走到头了。此时必须切 $S$、切 $V$，或者改变 loss 计算方式。

### 7.6 避免不必要的完整 logits 返回

一些训练代码为了方便 logging、metric、distillation 或 auxiliary loss，会把完整 logits 返回到上层。对于长序列训练，这很危险。更好的方式是：

- 只返回 loss，不返回完整 logits；
- metric 在 chunk 内计算；
- 如果只需要最后若干 token 的 logits，就不要保留全序列 logits；
- distillation 场景下考虑 top-k logits、chunked KL 或 vocabulary partition。

很多 OOM 并不是模型本身不可训练，而是训练 step 的输出接口太“宽”。

---

## 八、一个判断准则

判断某种并行策略能不能解决 OOM，可以问一个问题：

**造成 OOM 的那个最大张量，是否真的被这种并行策略切开了？**

如果答案是否定的，加再多 GPU 也只是让更多 GPU 各自 OOM。

更具体地说：

- ZeRO / FSDP 切的是模型状态，不自动切 logits；
- DP 切的是样本，不切单个样本内部的 sequence 或 vocab；
- TP 切的是 tensor 维度，但 logits 需要 vocab-parallel loss 才真正省显存；
- SP / CP 切的是 sequence / context，但 loss 前不能重新 gather；
- 加节点只有在扩大有效 shard 维度时才降低单卡显存；
- kernel / loss 实现不支持分块或分布式时，理论并行度不会自动变成实际显存收益。

训练 OOM 的排查重点，不是先问“我用了几张卡”，而是先问“峰值张量是什么形状、在哪个 rank 上被 materialize、它的哪个维度被切了”。

---

## 九、总结

大模型训练中的 OOM 可以分成两类：

第一类是**模型状态型 OOM**：参数、梯度、optimizer states 太大。这类问题适合用 ZeRO、FSDP、TP、PP 和更多 shard 来解决。

第二类是**局部计算型 OOM**：activation、attention scores、logits、loss backward buffer、workspace 太大。这类问题必须看具体张量形状。对于长序列训练尤其要警惕 $S^2$ 的 attention 相关张量和 $S \times V$ 的 logits / loss 相关张量。

开头提到的长序列 logits OOM，本质上属于第二类。它不会因为 ZeRO-3 或增加 DP 节点自然消失。真正有效的方向是：

- 用 vocab-parallel LM Head 和 cross entropy 切 $V$；
- 用 sequence-sharded loss 或 CP/SP 切 $S$；
- 用 chunked / fused cross entropy 控制 logits 峰值；
- 降低 micro batch，并用 gradient accumulation 保持全局 batch；
- 避免训练 step 返回或保存完整 logits。

并行训练不是“卡越多显存越小”，而是“被切开的那个维度，必须正好是造成 OOM 的那个维度”。只有把 OOM 的张量形状和并行策略的切分维度对齐，扩卡和并行才真正有意义。
