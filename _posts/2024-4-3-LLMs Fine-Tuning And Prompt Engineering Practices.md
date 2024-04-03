---

layout: post
tags: [Engineering, LLM]
title: LLMs Fine-Tuning And Prompt Engineering Practices
date: 2024-04-3
author: Kaka Chen
comments: true
toc: true
pinned: false

---

由于LLM的通用性，现在的LLM已经可以无需进行调整（例如，零样本学习）就直接用于某些特定领域的任务。 然而，为了能在特定领域的下游任务上获得更高的准确性，我们通常需要执行Fine-Tuning或Prompt Engineering以使得LLM来更好地适应下游任务。

---

# Fine-Tuning VS Prompt

Fine-Tuning是将pre-trained Model适应于特定任务或者特定领域的过程，比如执行分类或回答问题。对于一些LLMs而言，可能会考虑在产品文档上进行Fine-Tuning，或者在内部的客户服务消息语料库上对LLM进行Fine-Tuning，以帮助筛选最重要的请求。在这个过程中，通常会通过梯度下降的方法来更新模型全部或者部分的参数。但是在通常情况下，相较于pre-train model，fine-tuning model需要的数据要少得多（不到百分之一）。当然，如果是在一个原始LLM训练集中没有太多相似数据的领域，就可能需要更多的数据来达到你所期望的性能。

- Full fine-tuning（or simply ”Fine-Tuning“）：模型所有的参数都会被更新，相关的子类方法包括：
  - Transfer learning：更新pre-trained模型的一些层layers来适应一个新任务；
  - Knowledge Distillation：知识蒸馏，通常使用一个更小的成为”student“的模型去学习一个较大的模型”teacher“的表征，来应对新任务。
- Parameter-efficient fine-tuning：其余参数被冻结，只有一小部分参数更新。
  - Adapter-Tuning：在pre-trained LLM的层之间插入一些额外的跟任务相关的层，并且只微调这些新插入的层（adapter）的参数；
  - LoRA：其中的Adapter是原始权重矩阵的低阶近似（low rank approximations）；
  - Prefix-tuning：将特定于任务的向量添加到模型中，并且仅调整前缀中的参数。
- Instruction-tuning：使用监督样本进行微调，特别是作为指令表述的任务集合。所有模型参数都会被更新。它大幅提高了未见过的任务上的zero-shot性能，被认为是 2022 年 NLP 的主要突破之一。
- Reinforcement learning through human feedback（RLHF）：本质上是Instruction-tuning的扩展，在tuning之后加入更多新步骤。后续步骤的重点是确保模型符合人类偏好（真实性、毒性降低等）。
- Direct preference optimization（DPO）：一种稳定、高性能、计算量轻且简单得多的实现，使 LM 与人类偏好保持一致。

**Prompting**：也成为in-context learnig，涉及预先准备指令或为任务输入提供一些示例，并将这些示例提供给pre-trained LLMs。在这里，与其通过数百（或数千）个输入文本来微调模型，不如说模型通过接收一条指令和少量特定任务的示例（通常少于十个）来操作。这些示例（本质上展示了我们想要执行的任务），为模型提供了处理这类任务的指导蓝图，使其能够迅速适应并在新任务上有效地执行。在这个过程中，预训练模型的权重被冻结。相反，提示被设计和优化以引出来自大型语言模型的期望输出。换句话说，这里，你是通过指令和自然语言进行微调，而不是改变底层模型参数。

Prompt optimization类型：

- Prompt Engineering：设计提示模板的过程，该模板在下游任务上产生最有效的性能。提示工程可以通过手动或通过算法搜索来完成（不调整参数）。
- Prompt-Tuning：Prompt-Tuning在提示中引入了额外的参数，并使用监督样本优化这些参数。与在离散空间优化Prompt的Prompt Engineering相比，Prompt-Tuning编码了文本提示，并在连续空间中对其进行优化。

如何选择fine-tunig和prompt：这两种方法并不是互斥的，通常可以先用指令Fine-Tuning LLMs，使得其有较好的通用性能，再在交互的时候通过prompting来提升对特定下游任务的性能。

- 下游任务类型：对于特定有可客观正确问题答案的结构化任务（如分类、翻译），fine-tuning在低成本小模型上效果比prompt好；而对于创造性、生成性的，具有广泛的生成空间的任务（如仿照写作、代码补全、chatGPT类型的生成），prompt效果更好。
- 泛化性generalization和专业性specialization：Fine-tuning比prompt让LM有更好的条件反应。因此，fine-tuning通常能提高专业化能力，但代价是损害泛化能力。 如果你只有少数几个专业任务，fine-tuning可能会为每项任务提供最高的准确度，或者用更小的模型达到所需的准确度（这样可以降低推理成本）。 如果你有大量的下游任务，或者希望将来使用语言模型进行未来任务，efficient fine-tuning和prompt engineering允许你在不损害预训练大型语言模型（LLMs）的泛化能力的情况下，为模型提供专业化的条件。
- 可扩展性scalability和简单性simplicity：如果想要构建大型语言模型（LLMs）并将它们作为服务提供给广泛的内部或外部客户，采用大型语言模型并对每个下游任务进行efficient fine-tuning和prompt engineering是更具可扩展性的。 如果您是一个有限的使用场景集合的单一企业，采用一个较小的语言模型并将其fine-tuning以适应每个下游任务会更简单、更经济。
- 特定任务监督样本尺寸：如果有大量特定任务的监督样本，fine-tuning是可行的，并且会给你带来高度任务特定的准确性。如果事先很难收集特定任务的样本，或者一个任务的输入分布很广，fine-tuning变得不可行，而prompt engineering为zero-shot或few-shot learning提供了最佳性能。（例如，对于假新闻检测的用例，对于尚未发生的事件收集数据是不可能的）
- 推理的内存和存储成本：为了降低大量下游任务的推理成本，可以使用pre-trained LMs并对每个任务进行prefix-tuning或prompt tuning，或者使用较小的语言模型并为每个任务fully fine-tuning一个模型（尽管这可能因为适应所有下游任务的完整模型参数而变得成本过高）。尽管prompt engineering更直观，但可能会因为它涉及处理包含一个或多个训练示例的prompt来进行每次预测而变得昂贵。

**总结**

总的来说，鉴于固定语言模型（LM）大小的前提下，fine-tuning对于专门的任务而言是更优的，因为整个网络专门用来解决一个问题。然而，如果pre-trained LM足够大，且prompt足够好以至于能够“激活”大型网络中的正确权重，那么性能可以是可以相提并论的。从某种意义上说，你可以将prompt视为帮助通用网络表现得像专家网络的一种方式。

采用更实用的方法来使用大型语言模型（LLMs）涉及到将任务分解成组件，原型制作，并评估整个系统（外在的）及其组件（内在的）。这种方法经常被证明更有效且更易于维护，并提供了一种平衡的方式来利用LMs的能力，而无需高昂的运营成本和维护复杂性。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm_practice.png)

