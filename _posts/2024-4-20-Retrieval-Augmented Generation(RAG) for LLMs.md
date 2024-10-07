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

# 1 - 原理

RAG ，检索增强生成技术（Retrieval-Augmented Generation，RAG），通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG 有效地**缓解了幻觉问题**，提高了**知识更新的速度**，并增强了内容生成的**可追溯性**，使得大型语言模型在实际应用中变得更加**实用和可信**。

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

- **Ingestion（摄入）**：对每个文件做split到到多个文本chunks中，对于每一个chunk，根据embedding model进行text embedding，然后将每个embedding offload到一个index，这个index是存储系统的视图，例如vector database。
- **Retrieval（检索）**：根据生成的index启动查询，然后提取与查询最相似的top k个chunk
- **Synthesis（合成）**：将得到的top k chunks与用户查询合并，将其放入合成截断的LLM prompt window中，从而生成最终的结果内容。

改进方法：

- Sentence-window retrieval：Embedding and retrieving single sentence，在检索之后，句子将被替换为围绕原始检索句子的更大窗口的句子。
- Auto-Merging retrieval：构建树状chunk节点结构，将检索节点合并到较大的父节点中，意味着在检索过程中一个父节点有大多数子节点已检索，那么用父节点替换子节点。

## 2.2 - 进阶的 RAG（Advanced RAG）

Advanced RAG 范式增加了 `预检索` 、 `检索后处理` 等步骤  。在检索前阶段则可以使用**问题的重写**、**路由**和**扩充**等方式对齐问题和文档块之间的语义差异。在检索后阶段则可以通过将检索出来的文档库进行**重排序rerank**避免 “Lost in the Middle ” 现象的发生。或是通过上下文筛选与压缩的方式缩短窗口长度。

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
    1. Token：KNN-LMM擅长处理长尾和跨域问题，计算效率高，但需要大量存储；
    2. Phrase；
    3. Chunk：非结构化数据，召回大量信息但准确率低，存在较多冗余信息；
    4. Paragraph；
    5. Entity；
    6. Knowledge graph：丰富的语义和结构化数据，检索效率低，效果严重依赖KG本身质量
2. 何时检索：
    1. Single research：只检索一次，效率高，但可能导致检索结果相关度低；
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
        - Small-2-Big 在sentense级别做embedding
        - Slidingwindow 滑动窗口，让chunk覆盖整个文本，避免语义歧义
        - Summary 通过摘要检索文档，然后从文档中检索文本块。
    2. 可以添加一些额外的meta信息，例如page，时间，类型，文档标题等，并依此进行过滤或者增强信息量（Pseudo Metadata、Metadata filter）
2. 结构化检索文档库：可以分层组织检索文档库。
    1. Summary → Document方法， 用摘要检索取代文档检索，不仅可以检索最直接相关的节点，还可以探索与这些节点相关的其他节点
    2. Document → Embedded Objects 比如一个PDF文档具有嵌入对象（如表、图表），首先检索实体引用对象，然后查询底层对象，如文档块、[数据库](https://cloud.tencent.com/solution/database?from_column=20065&from=20065)、子节点
3. Knowledge Graph作为召回数据源：GraphRAG  从用户的输入查询中提取**实体**，然后**构建子图**以形成上下文，并最终将其输入到大模型中进行生成。
    1. 使用LLM 从问题中提取关键entity
    2. 基于提取的到entity实体，检索子图，并深入到一定的深度，比如2跳或者更多
    3. 利用获得的上下文通过LLM生成答案
4. Query优化：利用**Query Rewriting**和**Query Clarification**技术对原始query进行改写或者澄清，以获得最高的语义相似性，得到最佳的检索效果。
    - Query改写： 将query改写成一个或者多个search query，分别查询，这样可以得到更佳的召回效果。
    - Query澄清：用树形结构解释query中每一个语义点。
5. Embedding嵌入模型优化：选择合适的商用embedding或者自己微调embedding模型，其中微调可以通过领域数据和下游任务需要去微调。
6. 检索流程优化：可以有Iterative迭代式检索和Adaptive自适应检索。
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

RAG优点：

- **可扩展性 (Scalability)**：减少模型大小和训练成本，并允许轻松扩展知识
- **准确性 (Accuracy)**：模型基于事实并减少幻觉
- **可控性 (Controllability)**：允许更新或定制知识
- **可解释性 (Interpretability)**：检索到的项目作为模型预测中来源的参考
- **多功能性 (Versatility)**：RAG 可以针对多种任务进行微调和定制，包括QA、文本摘要、对话系统等。

# 8 - RAG技术生态

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240420-8.png)
