---

layout: post
tags: [LLM, NLP, RAG]
title: Retrieval-Augmented Generation(RAG) for LLMs
date: 2024-04-20
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

RAG ，检索增强生成技术（Retrieval-Augmented Generation，RAG），通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG 有效地**缓解了幻觉问题**，提高了**知识更新的速度**，并增强了内容生成的**可追溯性**，使得大型语言模型在实际应用中变得更加**实用和可信**。

# 1 - 前言

解决问题：

- **LLM缺乏特定领域知识；**
- **易产生幻觉；**
- **参数化知识效率低；**
- **存在过时信息；**
- **推理能力弱；**
- **更新模型所需的巨大计算资源**

RAG适用场景：

- 长尾数据；
- 频繁更新的知识；
- 需要验证和可追溯性的答案；
- 专业领域的知识；
- 数据隐私保护。
  
![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-1.png)

# 2 - RAG方法概述

## 2.1 - 朴素RAG（Naive RAG）

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-2.png)

- **Ingestion（摄入）**：对每个文件做 split 到多个文本 chunks 中；对每个 chunk 使用 embedding model 计算 embedding，并将 embedding 写入 index（例如 vector database）。
- **Retrieval（检索）**：根据生成的 index 启动查询，提取与查询最相似的 top‑k 个 chunk。
- **Synthesis（合成）**：将得到的 top‑k chunks 与用户查询合并，放入 LLM 的 prompt window 中生成最终答案。

改进方法：

- Sentence‑window retrieval：对单句做 embedding 并检索，命中后用该句周围更大窗口替换，以引入上下文。
- Auto‑Merging retrieval：构建树状 chunk 节点结构；当父节点的多数子节点被命中时，用父节点替换子节点，减少冗余。

## 2.2 - 进阶的 RAG（Advanced RAG）

Advanced RAG 范式增加了 `预检索` 、 `检索后处理` 等步骤。在检索前阶段可以使用**问题重写**、**路由**和**扩充**等方式对齐问题与文档块之间的语义差异；在检索后阶段可以通过对候选文档进行**重排序（rerank）**以避免 “Lost in the Middle” 现象，或通过上下文筛选与压缩缩短窗口长度。

## 2.3 - 模块化 RAG（Modular RAG）

模块化 RAG 在结构上它更加自由的和灵活，引入了更多的具体功能模块，例如**查询搜索引擎**、**融合多个回答**。技术上将检索与微调、强化学习等技术融合。流程上也对 RAG 模块之间进行设计和编排，出现了多种的 RAG 模式。然而，模块化 RAG 并不是突然出现的，三个范式之间是继承与发展的关系。Advanced RAG 是 Modular RAG 的一种特例形式，而 Naive RAG 则是 Advanced RAG 的一种特例。

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-3.png)

# 3 - 检索增强

RAG 系统中主要包含三个核心部分，分别是 “检索”，“增强” 和 “生成”。正好也对应的 RAG 中的三个首字母。想要构建一个好的 RAG 系统，增强部分是核心，则需要考虑三个关键问题：**检索什么？什么时候检索？怎么用检索的内容？**

**检索增强的阶段**：在预训练、微调和推理三个阶段中都可以进行检索增强，这决定了外部知识参数化程度的高低，对应所需要的计算资源也不同。

**检索增强的数据源**：增强可以采用多种形式的数据，包括非结构化的文本数据，如文本段落、短语或单个词汇。此外，也可以利用结构化数据，比如带有索引的文档、三元组数据或子图。另一种途径是不依赖外部信息源，而是充分发挥 LLMs 的内在能力，从模型自身生成的内容中检索。

**检索增强的过程**：最初的检索是一次性过程，在 RAG 发展过程中逐渐出现了迭代检索、递归检索以及交由 LLMs 自行判断检索时刻的自适应检索方法。

# 4 - RAG关键问题

1. 检索什么：
    1. Token：KNN‑LM 擅长处理长尾和跨域问题，计算效率高，但需要大量存储；
    2. Phrase；
    3. Chunk：非结构化数据，召回大量信息但准确率低，存在较多冗余信息；
    4. Paragraph；
    5. Entity；
    6. Knowledge graph：丰富的语义和结构化数据，检索效率低，效果严重依赖 KG 本身质量。
2. 何时检索：
    1. Single search：只检索一次，效率高，但可能导致检索结果相关度低；
    2. Each token；
    3. Every N tokens：会导致检索次数过多，并召回大量冗余信息；
    4. Adaptive search：平衡效率和检索效果。
3. 如何使用检索到的信息：
    1. Input/Data Layer：使用简单，但是无法支持更多knowledge blocks的检索，优化空间受限；
    2. Model/Intermediate Layer：支持更多knowledge blocks的检索，但是引入了额外复杂度，且必须经过训练；
    3. Output/ Prediction Layer：保证输出结果和检索内容高度相关，但是效率比较低。
4. 其他问题：
    - 论证阶段：Pre-Training、Fine-Tuning、Inference
    - 检索选择：BERT、Roberta、BGE…
    - 生成选择：GPT、Llama、T5…

# 5 - RAG关键技术：

1. 数据索引优化：
    1. 核心是chunk策略：
        - Small‑to‑Big：在 sentence 级别做 embedding
        - Sliding Window：滑动窗口，让 chunk 覆盖整个文本，避免语义歧义
        - Summary：通过摘要检索文档，然后从文档中检索文本块。
    2. 可以添加一些额外的 meta 信息，例如 page、时间、类型、文档标题等，并据此过滤或增强信息量（Pseudo Metadata、Metadata filter）。
2. 结构化检索文档库：可以分层组织检索文档库。
    1. Summary → Document 方法：用摘要检索取代文档检索，不仅可以检索最直接相关的节点，还可以探索与这些节点相关的其他节点
    2. Document → Embedded Objects：比如一个 PDF 文档具有嵌入对象（如表、图表），首先检索实体引用对象，然后查询底层对象，如文档块、数据库、子节点
 3. Knowledge Graph 作为召回数据源：GraphRAG 从用户输入中提取**实体**，然后**构建子图**以形成上下文，最终输入到大模型进行生成。
    1. 使用 LLM 从问题中提取关键实体；
    2. 基于提取到的实体检索子图，并深入到一定的深度（如 2 跳或更多）；
    3. 利用获得的上下文通过 LLM 生成答案。
 4. Query 优化：利用 **Query Rewriting** 和 **Query Clarification** 对原始 query 改写或澄清，以获得更高的语义相似性与召回质量。
    - Query 改写：将 query 改写成一个或多个 search query，分别查询，提升召回；
    - Query 澄清：用树形结构解释 query 中每个语义点。
 5. Embedding 嵌入模型优化：选择合适的商用 embedding 或微调自有 embedding 模型（用领域数据与下游任务目标做监督/对比学习）。
 6. 检索流程优化：可采用迭代式（Iterative）与自适应（Adaptive）检索。
7. Hybrid (RAG+Fine-tuning) 融合RAG和FT：既可以检索FT，也可以生成FT，还可以进行检索，生成联合FT

# 6 - 和微调对比

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-4.png)

RAG 就像给模型一本教科书，用于定制的信息检索，非常适合特定的查询。另一方面，FT 就像一个学生随着时间的推移内化知识，更适合模仿特定的结构、风格或格式。FT 可以通过增强基础模型知识、调整输出和教授复杂指令来提高模型的性能和效率。然而，它不那么擅长整合新知识或快速迭代新的用例。RAG 和 FT，并不是相互排斥的，它们可以是互补的，联合使用可能会产生最佳性能。

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-5.png)

# 7 - 如何评价RAG

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-6.png)

RAG 的评估方法多样，主要包括三个质量评分：**上下文相关性、答案忠实性和答案相关性**。此外，评估还涉及四个关键能力：噪声鲁棒性、拒答能力、信息整合和反事实鲁棒性。这些评估维度结合了传统量化指标和针对 RAG 特性的专门评估标准，尽管这些标准尚未统一。

1. **Context Relevance**：上下文相关性，评估生成的文本与检索到的上下文之间的相关性。
2. **Answer Faithfulness**：答案忠实度，确保生成的答案忠实于检索到的上下文，保持一致性。
3. **Answer Relevance**：答案相关性，要求生成的答案直接相关于提出的问题，有效解决问题。
4. **Accuracy**：准确性，衡量生成的信息的准确性。

在评估框架方面，存在如 RGB 和 RECALL 这样的基准测试，以及 RAGAS、ARES 和 TruLens 等自动化评估工具，它们有助于全面衡量 RAG 模型的表现。表中汇总了如何将传统量化指标应用于 RAG 评估以及各种 RAG 评估框架的评估内容，包括评估的对象、维度和指标，为深入理解 RAG 模型的性能和潜在应用提供了宝贵信息

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-7.png)

RAG 优点：

- **可扩展性 (Scalability)**：减少模型大小和训练成本，并允许轻松扩展知识
- **准确性 (Accuracy)**：模型基于事实并减少幻觉
- **可控性 (Controllability)**：允许更新或定制知识
- **可解释性 (Interpretability)**：检索到的项目作为模型预测中来源的参考
- **多功能性 (Versatility)**：RAG 可以针对多种任务进行微调和定制，包括QA、文本摘要、对话系统等。

# 8 - RAG 技术生态

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-8.png)

# 9 - 落地架构与性能工程

一个稳定的 RAG 系统更像一条“检索—重排—压缩—生成”的流水线。服务端通常通过批处理、并发与缓存来降低时延。向量检索层面可以为热点集合建立多级缓存（ANN 结果缓存、倒排/稠密混合预计算）；重排阶段需注意模型大小与吞吐的权衡（轻量 cross‑encoder 或列表截断）；压缩阶段要在“信息保真”与“窗口长度”间平衡。生成侧，合理的并发与 KV Cache 复用可以显著降低端到端时延。

# 10 - 提示工程与答案组织

提示设计直接决定“检索到的信息如何被使用”。常见有效做法包括：显式要求“仅基于上下文作答/不足则拒答”；要求逐条引用来源并在末尾给出引用列表；先抽取与聚合，再生成最终答案（extract‑then‑generate）；在多段来源矛盾时，要求模型标注冲突并优先更高置信来源。对需求复杂的问题，Map‑Reduce 与 Router‑expert 模式能减少一次性上下文挤压带来的丢失。

# 11 - 常见陷阱与排错

“Lost in the Middle” 与“Query Drift”是两类高频问题。前者可通过重排与窗口分配策略缓解；后者需在改写阶段对 query 做语义保持约束。此外，embedding 漂移（模型升级）会导致召回不一致，需做离线对比与兼容迁移；归一化/去重不充分会引入重复证据，放大偏见。建议上线“检索—重排—生成”各层可观测指标（召回@k、重排 NDCG、窗口利用率、拒答率、引用完整性），并做失败样例库。

# 12 - 隐私合规与数据治理

针对私域与敏感数据，RAG 的安全边界需要在摄入与生成两侧共同治理。摄入侧应做文档级与字段级脱敏（PII/PHI），并在索引中携带访问控制标签（ABAC/RBAC）；检索侧做策略过滤与审计；生成侧对输出做敏感信息检测与引用一致性校验。对跨域与外部检索，应避免把用户上下文泄露给第三方服务，或采用代理检索与最小化传输。

# 13 - 结语

RAG 的核心价值在于让“外部知识”以结构化的方式进入生成过程。把“检索—重排—压缩—生成”这条链路打通，并以可靠的评测与可观测性持续迭代，才能在真实业务场景里获得稳定、可解释、可演进的效果。
