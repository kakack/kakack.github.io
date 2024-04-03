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

由于LLM的通用性，现在的LLM已经可以无需进行fine-tuning（例如，零样本学习）就直接用于某些特定领域的任务。 然而，为了能在特定领域的下游任务上获得更高的准确性，我们通常需要执行Fine-Tuning或Prompt Engineering以使得LLM来更好地适应下游任务。

---

# Fine-Tuning VS Prompt

Fine-Tuning是将pre-trained Model适应于特定任务或者特定领域的过程，比如执行分类或回答问题。对于一些LLMs而言，可能会考虑在产品文档上进行Fine-Tuning，或者在内部的客户服务消息语料库上对LLM进行Fine-Tuning，以帮助筛选最重要的请求。在这个过程中，通常会通过梯度下降的方法来更新模型全部或者部分的参数。但是在通常情况下，相较于pre-train model，fine-tuning model需要的数据要少得多（不到百分之一）。当然，如果是在一个原始LLM训练集中没有太多相似数据的领域，就可能需要更多的数据来达到你所期望的性能。

- Full fine-tuning（or simply ”Fine-Tuning“）：模型所有的参数都会被更新，相关的子类方法包括：
  - Transfer learning：更新pre-trained模型的一些层layers来适应一个新任务；
  - Knowledge Distillation：知识蒸馏，通常使用一个更小的成为”student“的模型去学习一个较大的模型”teacher“的表征，来应对新任务。
- Parameter-efficient fine-tuning：其余参数被冻结，只有一小部分参数更新。
  - Adapter-Tuning：在pre-trained LLM的层之间插入一些额外的跟任务相关的层，并且只fine-tuning这些新插入的层（adapter）的参数；
  - LoRA：其中的Adapter是原始权重矩阵的低阶近似（low rank approximations）；
  - Prefix-tuning：将特定于任务的向量添加到模型中，并且仅fine-tuning前缀中的参数。
- Instruction-tuning：使用监督样本进行fine-tuning，特别是作为指令表述的任务集合。所有模型参数都会被更新。它大幅提高了未见过的任务上的zero-shot性能，被认为是 2022 年 NLP 的主要突破之一。
- Reinforcement learning through human feedback（RLHF）：本质上是Instruction-tuning的扩展，在tuning之后加入更多新步骤。后续步骤的重点是确保模型符合人类偏好（真实性、毒性降低等）。
- Direct preference optimization（DPO）：一种稳定、高性能、计算量轻且简单得多的实现，使 LM 与人类偏好保持一致。

**Prompting**：也成为in-context learnig，涉及预先准备指令或为任务输入提供一些示例，并将这些示例提供给pre-trained LLMs。在这里，与其通过数百（或数千）个输入文本来fine-tuning模型，不如说模型通过接收一条指令和少量特定任务的示例（通常少于十个）来操作。这些示例（本质上展示了我们想要执行的任务），为模型提供了处理这类任务的指导蓝图，使其能够迅速适应并在新任务上有效地执行。在这个过程中，预训练模型的权重被冻结。相反，提示被设计和优化以引出来自大型语言模型的期望输出。换句话说，这里，你是通过指令和自然语言进行fine-tuning，而不是改变底层模型参数。

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

---

# Fine-Tuning LLMs

**Fully Fine-Tuning**：指取一个pre-trained model，并fine-tuning它以优化特定任务的过程。在fine-tuning过程中，通过梯度下降更新预训练模型参数。虽然fine-tuning的前提看似简单，但其执行可分为三个核心方法，每种方法都有其独特的程序和结果。

- Featured-based approach：在基于特征的方法中，我们使用pre-trained LM并将其应用于目标数据集，主要关注于为训练集生成输出嵌入output embedding。这些输出嵌入可以作为输入特征来训练分类模型，例如逻辑回归模型、随机森林或XBoost。尽管通常用于以嵌入为重点的模型，如BERT，我们也可以从生成式的GPT风格模型中提取嵌入；
- Updating the output layers：与Featured-based approach密切相关，冻结其他层的状态，但只fine-tuning输出层，例如一个分类层；
- Updating all layers：更新所有层，这种方法由于涉及的参数数量增加而更加昂贵，但通常会导致更优越的建模性能。例如，虽然一个BERT基础模型大约包含1.1亿个参数，但用于二分类的BERT基础模型的最后一层只包含约1500个参数。然而，一个BERT基础模型的最后两层大约占60000个参数——大约占总模型大小的0.6%。尽管性能会根据目标任务与模型预训练所用数据集之间的相似性而有所不同，但几乎总是通过fine-tuning所有层来提升模型的性能。

优劣对比：

- 优势：Fully fine-tuning后的大型语言模型（LLMs）如果在fine-tuning过程中有足够的监督样本，可能会达到下游任务的最高准确率。
- 劣势：
  - 这种技术产生了一个专门针对单一任务的模型，具有一整套全新的参数值，因此，如果有许多下游任务都需要训练自己的模型，这可能会变得非常昂贵。你可以将此与在相同的语言模型大小上使用参数efficient fine-tuning进行比较（以个性化为例用例）。
  - Fine-tuned model可能会在较小的数据集上过拟合或学习不稳定，意味着每个新任务都需要一个新的大数据集。
  - 与in-context learning或者prompt-based方法相比，fine-tuned models可能会缺少一些分布外的泛化能力（out-of-distribution generalization capabilities）。



**Parameter-efficient fine-tuning**：是fully fine-tuning的轻量级替代方案，它冻结了大部分预先训练的参数，并使用小型可训练模块增强模型。

- Adapter-tuning：Adapter-tuning 在预训练语言模型的各层之间插入额外的任务特定层，这些被称为适配器Adapter。在调整过程中，只有Adapter的参数会被调整，而预训练模型的参数则保持不变。LoRA（Low-Rank Adaptation）是最近流行起来的一种fine-tuning方法，它最小化了调整参数的数量，优化了调整内存效率，缓解了灾难性遗忘，同时不增加额外的推理延迟。LoRA在保持模型预训练权重不变的同时，增加了一对秩分解权重矩阵（称为更新矩阵）对，且只训练这些新添加的权重。
- Prefix-tuning：Prefix-tuning也是一种轻量化替代方案。它在模型前面添加一系列连续的任务特定向量，这被称为前缀Prefix。前缀完全由自由参数组成，这些参数不对应真实的tokens。在fine-tuning过程中，只有前缀参数被调整，而预训练模型参数保持冻结。这种方法的灵感来自于prompt。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm_practice2.png)

优劣对比：

- 优势：
  - 在相同的预训练大型语言模型（LLM）规模下，比fully fine-tuning更便宜。Parameter-efficient fine-tuning仅涉及模型全部参数的小部分，甚至个位数百分比，同时达到与fully fine-tuning相媲美的结果，导致任务特定参数减少10-1000倍。
  - Parameter-efficient fine-tuning，尤其是adapter-tuning和prefix-tuning，在高数据环境下（即监督样本的数量），可以获得与fully fine-tuning相媲美的性能，在低数据环境下性能更佳，并且对训练中未见过的主题示例有更好的外推能力。
  - Parameter-efficient fine-tuning在准确性上优于prompt engineering。此外，它比离散的prompt engineering更有表达性，后者需要匹配真实词的嵌入。
  - 尤其是prefix-tuning允许在推理时进行混合任务批处理（不同任务的查询可以在同一个批次中），因此在GPU共享方面具有计算效率（注：adapter-tuning的推理不能批处理）。
  - 一些研究显示，与adapter-tuning相比，prefix-tuning需要的参数数量显著减少，同时保持了可比较的性能。直觉是，prefix-tuning尽可能保持预训练的语言模型不变，因此比adapter-tuning更多地利用了语言模型。
- 劣势：
  - 在一些任务上，如大数据两的摘要生成任务等，可能比fully fine-tuning性能效果差。
  - 需要比prompt engineering更多的监督样本。

---

# Prompting

**Prompt-tuning**：和prefix-tuning很像，可以被认为是一个简化版。我们冻结了pre-trained model参数，仅允许每个下游任务额外添加一组可调整的tokens，这些tokens会被添加到输入文本之前。这种“soft prompt”可以压缩通常较大的监督样本中的信号，允许该方法超越少few-shot prompts，并缩小与模型fine-tuning的质量差距。同时，由于单个pre-trained model可以被重复用于所有下游任务，这种方法在需要服务的任务种类繁多且任务量大时，具有更好的扩展性。鉴于其与 prefix-tuning 的相似性，prompt-tuning 与 prefix-tuning 有着类似的优点和缺点。

**Instruction-tuning**：这是一种state-of-the-art的fine-tuning技术，它在一个以指令形式表达的任务集上对pre-trained LLMs进行监督式fine-tuning。它使得pre-trained LLMs能够更好地响应指令，并减少在Promp阶段（即显著改善零次示例性能）需要的少数样本示例的需求。

2022年，Instruction fine-tuning因为该技术在不损害模型的泛化能力的情况下显著提升模型性能而获得了巨大的流行。通常，一个pre-trained LM会在一系列语言任务上进行fine-tuning，并在fine-tuning时未见过的另一组语言任务上进行评估，以证明其泛化能力和zero-shot能力。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm_practice3.png)

优劣对比：

- 优势：
  - 2022年在语言模型（LM）领域的一个突破是，instruction fine-tuning成为了提高大型语言模型（LLMs）整体性能和有用性的state-of-the-art。
  - 特别是在零样本学习场景中（即面对未见过的任务时），instruction fine-tuning显著提高了模型的性能，因此经过Instruction fine-tuning的LLMs比那些更多为特定用例而精细fine-tuning的模型具有更强的泛化能力。
- 劣势：
  - 这种方法调整了整个模型的参数，而不是在paramater-efficient fine-tuning中冻结其中一部分。这意味着它不会带来参数高效fine-tuning所具有的成本优势。然而，鉴于instruction fine-tuning产生的模型相比parameter-efficient fine-tuning更具有泛化能力，instruction fine-tuning的模型仍然可以作为服务于多个下游任务的通用模型。本质上，这取决于你是否拥有进行instruction fine-tuning所需的指令数据集以及训练预算。
  - Instruction fine-tuning在自然以指令形式表述的任务上普遍有效（e.g., NLI, QA, translation, struct-to-text），但对于直接以语言建模形式构建的任务（例如，推理）就有点更棘手了。为了使instruction fine-tuning在推理任务中也能比传统fine-tuning表现得更好，需要在instruction fine-tuning的监督样本中包含chain-of-thought示例。

**Reinforcement learning through human feedback（RLHF）**：
RLHF 是instruction fine-tuning的扩展，在instruction fine-tuning步骤之后增加了更多步骤，以进一步纳入人类反馈。
预训练的大型语言模型（LLMs）经常表现出非预期的行为，如捏造事实（也称为幻觉）、生成有偏见或有害的回答，或简单地不遵循用户指令。这是因为许多最新的语言模型使用的语言建模目标——从互联网上的网页预测下一个词——与“有帮助且安全地遵循用户指令”的目标不同。
通过人类反馈进行强化学习是一种fine-tuning技术，帮助语言模型按照用户的明确意图（如遵循指令、保持真实）和期望行为（不表现出有偏见、有害或其他有害特征）行动。
这项技术的最佳示范是由OpenAI提供的。他们采用了他们的预训练GPT-3模型，使用RLHF进行了微调，并派生出他们称为InstructGPT的更加对齐的模型。此外，ChatGPT也是使用RLHF在更高级的GPT模型系列（称为GPT-3.5）上派生出来的。

RLHF步骤：
1. Instruction-tuning：收集期待模型完成的行为并以此通过监督学习fine-tuning LLMs；
2. 收集模型输出之间比较的数据集，其中标记器指示他们对于给定输入更喜欢哪个输出。 然后，训练奖励模型来预测人类偏好的输出。
3. 采用经过训练的奖励模型，并使用强化学习针对奖励模型优化策略（step 2、3不断循环）。

**RLHF总结**：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm_practice4.png)

