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
   ![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-01.gif)
3. 筛选和确认：若猜测的 token 序列在大模型的 logits 确认下可信（即与大模型可能选择的高概率 token 一致），则直接接受整个草稿序列，从而节省了逐 token 生成的反复计算时间；若猜测不符，则丢弃并重新尝试。
    ![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-02.gif)

在贪婪解码情境下，推测解码之所以能够在理论上保留与大模型原本逐步生成相同的概率分布特征，其关键在于“确认”步骤的严格性。贪婪解码策略下，大模型在每个位置都会选择当前概率分布中最高概率的下一个 token。推测解码既没有改变大模型给出概率分布的方式，也没有在最终决策上偏离大模型的“贪婪”选择逻辑。它只是在过程中预先尝试一批潜在的后续 token，并使用大模型原本的概率分布为标准进行严格筛选。一旦筛选通过，就说明这些 token 本来就是大模型在逐 token 解码中所可能给出的最高概率路径，从而保证最终输出与传统自回归解码的结果在理论上是一致的。这一过程并未更改大模型本来的输出分布，只是通过提前拟合和后验验证的方式，加快了确认下一个最优 token 的决策速度。

## Eagle

Eagle 是 Google 的一个开源项目，它利用了 Speculative Decoding 的思想，实现了一种基于大模型 Drafting 的方法，以加速大模型的解码速度。它的创新点在于，不直接对下一步 token 进行预测，而是对原始大模型内部的最后一个隐藏层特征（来自 lm_head 前面一层的 feature）进行外推（extrapolate）。

EAGLE 引入了一个轻量级的自回归头（Auto-regression Head），基于当前特征序列预测下一个特征，最后通过冻结的分类头将 features 映射为 tokens。这种方法的优势在于，features 比 tokens 更具结构性和规律性，因此能达到更好的草稿接受率。此外，EAGLE在起草阶段采用树状生成结构，使得在验证阶段可以通过一次前向传播处理多个 tokens，从而提高了解码效率。

- EAGLE 与 SpecInfer 和 Medusa 类似，采用树注意力机制，草稿的生成和验证都是树结构的。
- 需要一个基础大模型和一个附加模块（FeatExtrapolator），这个 FeatExtrapolator模块的参数量远小于大模型，例如70B的大模型对应1B的 FeatExtrapolator。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-03.png)

与其他方法只基于 token 进行起草不同，EAGLE 还基于 feature 序列（f4, f5）进行起草。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-04.png)

具体流程如下：

1. 初始前向计算（原模型生成初始特征和下一个token）：
首先使用原始LLM对给定的输入prompt进行一次标准的自回归前向传播，得到下一 token 的特征表示（第二高层特征）和对应的token输出。这一步是正常解码的开始点，得到初始的已知特征和token。
2. 特征外推（起草）：
利用上述已获得的特征和当前 token 的词嵌入，输入到轻量的 Auto-regression Head（FeatExtrapolator）中进行预测。该头通过自回归方式生成后续的特征序列（即对下一步、下下步的特征进行外推预测）。
在得到这些预测特征后，使用冻结的LM头将这些特征映射回 token 分布，并根据该分布进行采样，得到多条可能的token序列分支。最后在特征层面快速生成了一个树形的候选token集。
3. 多轮推测与树状生成：
重复上述特征外推和token生成的过程多次，即在每一轮中，从当前已验证的token和特征出发，通过 FeatExtrapolator 再猜测多个后续特征点，并通过 LM头生成多个候选 token 分支，形成一颗较为稀疏的预测树。这一过程仅使用小模型（FeatExtrapolator）快速起草出大量候选序列。
4. 验证（单次前向评估原LLM）：
对通过树状起草的候选序列进行验证，即使用原始LLM进行一次前向传播，验证这些猜测路径中哪些分支的token是与原LLM分布一致的，并选出要接受的token。

## Eagle-2

相比EAGLE-1的改进：

- 在 EAGLE-1 中使用静态 draft 树，这假设 draft tokens 的接受率仅取决于它们的位置。但是现在，我们发现 draft tokens 的接受率还取决于上下文。因此 EAGLE-2 使用了一种具有上下文感知的动态draft树。
- 回归了传统推测采样（speculative sampling）方法的假设：根据上下文的变化，某些 tokens 更简单，更容易通过较小的草稿模型预测。

EAGLE-2 在 EAGLE 的基础上，通过 扩展（Expand）和 重排序（Rerank）两个阶段实现对 draft 树的动态调整，实现了对生成过程的进一步优化和加速，提高了对高价值分支的优先性，从而在验证前就大幅减少了需要处理的冗余候选序列。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-05.png)

1. 扩展阶段（Expansion Phase）：
   - 在当前层中，根据全局接受率（通过节点接受率与置信度近似计算得到的值）对节点进行排名。
   - 选取 top-k 个具有最高全局接受概率的节点作为输入，再次输入给起草模型进行下一层候选 token 的生成，从而扩展起草树。这样就避免了过度扩张，减少了单次前向计算的负担。
2. 重排序阶段（Reranking Phase）：
   - 在扩张完成后，对所有已生成的候选 token 节点（包括浅层和深层）进行全局排序，选取 top-m 个最有可能被接受的 token。
   - 对于同值节点优先选浅层节点，以确保选出的前 m 个 token 仍然保持一个树状结构。
   - 将这些选中的节点线性化成一条序列，以作为下一步验证阶段的输入。

Attention Mask 的调整。因为最终输入给 LLM 验证的序列来自一棵树的不同分支，这些分支不应共享上下文。通过调整注意力掩码，只让每个 token 看见它的祖先节点，保证生成和原始自回归过程在信息可见性上的一致性。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-06.png)

# 大模型draft+verify

## Medusa

Medusa 是一种不需要小草稿模型，而是通过扩展大模型结构本身来实现推测解码的方法。它在大模型中添加多个额外的并行解码头 "Medusa heads"，这些解码头与原始模型无缝集成，并且使用 tree-based attention 并行验证多个候选序列。Medusa Heads 需要经过微调，主干网络会被冻结。

### Medusa Heads

每个 Medusa Head 实现为单层前馈网络，并通过残差连接进行对齐。每个 Medusa Head 位于大模型的 lm_head 之前，接受 last hidden state 作为输入，输出预测的单个 token。在单次前向传播中，第一个 Medusa Head 预测输入序列的下一个（第i+1）token，第二个 Medusa Head 预测输入序列的偏移量为(i+2)个 token。

```python
self.medusa_head = nn.ModuleList([
   nn.Sequential(
      *([ResBlock(self.hidden_size)] * medusa_num_layers),
      nn.Linear(self.hidden_size, self.vocab_size, bias=False),
   )
   for _ in range(medusa_num_heads)
  ])
```

### Tree-based Attention

Medusa 使用 tree-based attention 并行验证多个候选序列。在每个 Medusa Head 中，使用 tree-based attention 来计算每个候选序列的 logits。Medusa 采用自上而下的方法构建树，对于每个 Medusa Head 生成的候选 token 作笛卡尔积（Cartesian product）：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-07.png)

在上图中，第一个 Medusa Head 的前 2 个预测 token (It, I) 和第二个 Medusa Head 的前 3 个预测 token (is, ’, the) 总共产生 2 × 3 = 6 个 candidates（候选序列），这些 candidates 中的每一个都对应树中的一个不同分支。

为了保证每个 Token 只访问其前驱，本文重新设计了 attention mask，它只允许 Attention 从当前 Token 流回其前驱 Token。位置编码的位置索引（position idx）会根据此结构进行调整。

### Sparse Tree

当每个 Medusa Head 取 top-k个（而不是top-1）草稿 token 时，树的路径数就会急剧增多，因此设计了一种稀疏化注意力树的方法。在实验中，具有 64 个节点的稀疏树比具有 256 个节点的密集树显示出更好的加速比。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/241010-08.png)

### Implementation

首先调用 [utils.py] 的 generate_medusa_buffers 函数，得到一个初始化的稀疏树结构，这个函数为每条路径创建一个祖先节点索引列表，以便在后续进行验证时能够快速找到与当前节点路径相关的所有祖先路径：

```python
retrieve_indices_nest = []
retrieve_paths = []
for i in range(len(sorted_medusa_choices)):
    cur_medusa_choice = sorted_medusa_choices[-i-1]  # 从列表末尾开始处理路径
    retrieve_indice = []
    if cur_medusa_choice in retrieve_paths:
        continue
    else:
        for c in range(len(cur_medusa_choice)):
            # 对于当前路径的每个前缀子路径（从 root 一直到当前深度的子路径），
            # 找出其在 sorted_medusa_choices 中的索引。
            subpath = cur_medusa_choice[:c+1]
            retrieve_indice.append(sorted_medusa_choices.index(subpath))
            retrieve_paths.append(subpath)
    retrieve_indices_nest.append(retrieve_indice)
```

- 逆序处理路径：cur_medusa_choice = sorted_medusa_choices[-i-1] 从列表的末尾往前取路径，保证了当处理深层节点时，其祖先路径也已处理过。
- if cur_medusa_choice in retrieve_paths: continue 用于对路径去重，避免重复处理相同路径。
- 构造单条路径的祖先索引列表：对 cur_medusa_choice 的每个前缀 cur_medusa_choice[:c+1]，找到该前缀路径在 sorted_medusa_choices 中的全局索引并加入 retrieve_indice。这样 retrieve_indice 就变成从根节点开始，一直到当前路径末尾的一个祖先链条的全局索引序列。

完成后 retrieve_indices_nest 是一个嵌套列表，每个子列表代表一条路径及其所有祖先路径在 sorted_medusa_choices 中的索引序列：

```python
max_length = max([len(x) for x in retrieve_indices_nest])
retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
retrieve_indices = retrieve_indices + 1
retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)
```

最终将生成的所有张量聚合到一个字典 medusa_buffers 中，返回这个 medusa_buffers。

然后回到 [medusa_model_new.py]，将 medusa_buffers 传入 generate_candidates 函数生成候选 candidates 序列，

```python
for idx in range(max_steps):
    # Generate candidates with topk predictions from Medusa heads
    candidates, tree_candidates = generate_candidates(
    medusa_logits,
    logits,
    medusa_buffers["tree_indices"],
    medusa_buffers["retrieve_indices"],
    …
)
```

得到候选序列后，再对其应用 tree_decoding 函数，tree_decoding 大致具有三个步骤：

1. 通过将 Medusa position IDs 添加到输入序列，计算新的位置 ID。
2. 使用大模型来 decode the tree candidates，模型返回原始 logit 和可能的其他输出的 logit。
3. 根据 retrieve_indices重新排序获得的 logit，以确保与 reference 顺序一致。

```python
# Use tree attention to verify the candidates and get predictions
medusa_logits, logits, outputs = tree_decoding(
    tree_candidates,
    past_key_values,
    medusa_buffers["medusa_position_ids"],
    medusa_buffers["retrieve_indices"],
)
```

### Conclusion

Medusa 真正吸引人的地方在于它上一步生成草稿，后续就可以直接把生成的草稿送进 lm_head 验证，在单次前向传播中同时做到草稿+验证，这一点看起来比 Draft & Verify 好，也避免了生成草稿阶段的推理延迟和显存占用问题。

文章更集中于测试其在放宽验证标准下（Typical Acceptance）的结果，这部分的结果无法保证模型的生成质量；

`我们确定通常没有必要匹配原始模型的分布。因此，我们建议采用 Typical Acceptance 方案来选择合理的candidates`

在贪婪解码下，虽然接受率很低（0.6），但从文章的测试结果来看，整体加速比也能达到不错的效果，这可能主要来源于整体草稿+验证速度的提升。

## Hydra

Hydra 是对Medusa 的改进，它对 Medusa Heads 的结构进行了简单的更改。在 Medusa 中，所有Medusa Heads都是独立的，但 Hydra Heads 会将 candidate 序列中的早期 tokens 作为附加输入：

## Draft & Verify

## Lookahead Decoding

# 通过检索生成draft

## REST

## Prompt Lookup Decoding

# 与稀疏kv cache结合

## MagicDec

# 实验效果

---

# Reference

- [Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation](https://arxiv.org/abs/2203.16487)
- [Fast Inference from Transformers via Speculative Decoding, Yaniv Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling, Charlie Chen et al., 2023](https://arxiv.org/abs/2302.01318)
- [Eagle](https://github.com/SafeAILab/EAGLE)

