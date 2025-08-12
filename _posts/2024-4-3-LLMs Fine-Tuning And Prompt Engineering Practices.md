---

layout: post
tags: [Engineering, LLM]
title: LLMs Fine-Tuning And Prompt Engineering Practices
date: 2024-04-03
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

由于 LLM 的通用性，很多场景可以无需进行 fine-tuning（如 zero-shot）就直接完成任务。然而，为了在特定领域的下游任务上获得更高的准确性与稳定性，通常需要执行 Fine-Tuning 或 Prompt Engineering，使 LLM 更好地适应目标分布。

---

# Fine-Tuning vs Prompt

Fine-Tuning 是将 pre-trained model 适配到特定任务/领域的过程（如分类、问答）。例如，可在产品文档或客服语料上 fine-tuning，以帮助更好地理解专业术语与格式约束。过程上通过梯度下降更新全部或部分参数。通常相较预训练阶段，fine-tuning 所需数据量要少得多（常低于 1%）。但若目标领域在预训练数据中覆盖较少，则可能需要更大的标注集才能达到期望性能。

- Full fine-tuning（或简称 Fine-Tuning）：模型所有参数都会被更新，相关变体包括：
  - Transfer learning：更新 pre-trained 模型的部分层以适应新任务；
  - Knowledge Distillation：知识蒸馏，使用更小的 student 学习较大 teacher 的表征/决策。
- Parameter-efficient fine-tuning（PEFT）：冻结大部分参数，仅更新少量新增/重参数化模块。
  - Adapter-tuning：在各层间插入任务相关的适配器，仅更新适配器参数；
  - LoRA/QLoRA：将原始权重以低秩近似增量形式更新（QLoRA 结合量化以节省显存）；
  - Prefix/Prompt-tuning：为注意力注入可学习前缀/软提示，仅更新前缀参数；
  - P-Tuning v2/IA3 等：更多轻量高效的参数注入方式。
- Instruction-tuning：在“指令+示例”形式的任务集合上做监督式 fine-tuning（通常更新全部参数），显著提升 zero-shot/few-shot 的指令遵循能力。
- Reinforcement learning through human feedback（RLHF）：本质上是Instruction-tuning的扩展，在tuning之后加入更多新步骤。后续步骤的重点是确保模型符合人类偏好（真实性、毒性降低等）。
- Direct preference optimization（DPO）：一种稳定、高性能、计算量轻且简单得多的实现，使 LM 与人类偏好保持一致。

**Prompting**：也称 in-context learning，指预先准备指令或为任务输入提供少量示例（few-shot），并与输入一同提供给预训练 LLM。在这里不更新模型参数，而是通过自然语言的“外部条件”引导模型行为，可视为“通过提示替代参数更新”。

Prompt 优化类型：

- Prompt Engineering：手工或自动搜索设计 prompt 模板（不调整模型参数）。
- Prompt-tuning：在 prompt 中引入可学习参数（软提示），用监督样本优化，属于连续空间优化。

如何选择 fine-tuning 和 prompt：二者并非互斥。常见做法是先做 Instruction-tuning 获得良好通用能力，再用 prompting/PEFT 面向具体任务做增益。

- 下游任务类型：客观可判定的结构化任务（如分类、翻译），小模型 + fine-tuning 往往更优；创造性/开放生成（仿写、代码补全、聊天）则更依赖 prompting 与大模型容量。
- 泛化性与专业性：fine-tuning 更有利于“条件反应”，通常提升专业化能力，但可能牺牲泛化。少量明确专业任务可倾向 fine-tuning；大量多样任务可优先大模型 + prompting/PEFT。
- 可扩展性与简单性：服务多任务/多租户时，大模型 + prompting/PEFT 更具扩展性；单一企业少数场景可选小模型 + fine-tuning，降低推理成本。
- 监督样本规模：样本充足时 fine-tuning 可得到高度任务特定的准确性；样本难收集或输入分布很广时，prompt engineering 更合适（zero-shot/few-shot）。
- 推理成本：为大量下游任务降本时，可用 prefix/prompt‑tuning 在共享底模上复用；或使用小模型 + fully fine-tuning（但需管理多套权重）。prompting 直观易用，但 few-shot 样例会增加上下文长度与推理成本。

**总结**

总的来说，鉴于固定语言模型（LM）大小的前提下，fine-tuning对于专门的任务而言是更优的，因为整个网络专门用来解决一个问题。然而，如果pre-trained LM足够大，且prompt足够好以至于能够“激活”大型网络中的正确权重，那么性能可以是可以相提并论的。从某种意义上说，你可以将prompt视为帮助通用网络表现得像专家网络的一种方式。

更实用的方法是将任务分解为组件，快速原型，并分别评估系统层面（外在）与组件层面（内在）表现，以在成本与可维护性间取得平衡。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm-practice.png)

---

# Fine-Tuning LLMs

**Fully Fine-Tuning**：取一个 pre-trained model，并 fine-tuning 以优化特定任务。可分为三类路径：

- Feature-based approach：使用 pre-trained LM 生成输出 embedding，作为特征训练下游分类器（如逻辑回归、随机森林或 XGBoost）。该方法常用于 BERT 类模型，也可从 GPT 风格模型提取嵌入；
- Updating the output layers：与上法相近，冻结主干，仅 fine-tuning 输出层（如分类头）；
- Updating all layers：更新所有层，成本更高但性能通常更好。尽管效果取决于下游任务与预训练数据的相似性，但“全量更新”往往带来最优效果。

优劣对比：

- 优势：Fully fine-tuning后的大型语言模型（LLMs）如果在fine-tuning过程中有足够的监督样本，可能会达到下游任务的最高准确率。
- 劣势：
  - 这种技术产生了一个专门针对单一任务的模型，具有一整套全新的参数值，因此，如果有许多下游任务都需要训练自己的模型，这可能会变得非常昂贵。你可以将此与在相同的语言模型大小上使用参数efficient fine-tuning进行比较（以个性化为例用例）。
  - Fine-tuned model可能会在较小的数据集上过拟合或学习不稳定，意味着每个新任务都需要一个新的大数据集。
  - 与in-context learning或者prompt-based方法相比，fine-tuned models可能会缺少一些分布外的泛化能力（out-of-distribution generalization capabilities）。

## 数据准备与质量治理（Fine-Tuning 前）

在实践中，“数据质量 > 算法细节”。建议：

- 标注与清洗：去除脏数据、泄露答案、PII/PHI；规范化格式（JSONL/指令-输出对），确保指令与答案一一对应；
- 去重与降噪：近重复（near-dup）去重、模板化内容去重，避免过拟合；
- 覆盖与难度：覆盖目标领域关键意图、边界条件与反例，按难度分层（easy/medium/hard），保证“可学习信号”；
- 安全与合规：对指令与输出做安全标注（拒答/转接），为后续对齐与守护（guardrail）打基础。

## 训练 Recipe（建议默认）

- 优化器：AdamW（β1=0.9, β2=0.95, eps=1e‑8），线性/余弦学习率，3–5% warmup；
- 学习率范围（参考 7B/13B）：全量 FT 2e‑5 ~ 5e‑5；PEFT（LoRA）5e‑4 ~ 1e‑3；
- 批量与累积：尽量提高有效 batch size，不足用梯度累积；
- 正则与稳定性：label smoothing（分类）、梯度裁剪（1.0）、混合精度（bf16/fp16）、梯度检查点；
- LoRA 配置：秩 r=8/16/32，α=16/32，target=[q_proj,k_proj,v_proj,o_proj,gate,up,down]；
- 训练时长：以验证集曲线/早停控制，避免过拟合；必要时混入小比例通用语料做“维持训练”。

## 避免遗忘与对齐退化

- Catastrophic Forgetting：混入少量通用语料（rehearsal）、KL 约束或 L2‑SP 正则；
- 格式与风格漂移：统一对齐模板与系统提示；加入“格式/风格”对照样本；
- 指标倒挂：同时跟踪内在指标（困惑度/准确率）与外在指标（延迟/吞吐/成本/拒答率）。


**Parameter-efficient fine-tuning**：是 fully fine-tuning 的轻量替代，冻结大部分参数，仅训练小型可学习模块。

- Adapter-tuning：Adapter-tuning 在预训练语言模型的各层之间插入额外的任务特定层，这些被称为适配器Adapter。在fine-tuning过程中，只有Adapter的参数会被更新，而预训练模型的参数则保持不变。LoRA（Low-Rank Adaptation）是最近流行起来的一种fine-tuning方法，它最小化了更新参数的数量，优化了调整内存效率，缓解了灾难性遗忘，同时不增加额外的推理延迟。LoRA在保持模型预训练权重不变的同时，增加了一对秩分解权重矩阵（称为更新矩阵）对，且只训练这些新添加的权重。
- Prefix-tuning：Prefix-tuning也是一种轻量化替代方案。它在模型前面添加一系列连续的任务特定向量，这被称为前缀Prefix。前缀完全由自由参数组成，这些参数不对应真实的tokens。在fine-tuning过程中，只有前缀参数被更新，而预训练模型参数保持冻结。这种方法的灵感来自于prompt。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm-practice2.png)

优劣对比：

- 优势：
  - 在相同的预训练大型语言模型（LLM）规模下，比fully fine-tuning更便宜。Parameter-efficient fine-tuning仅涉及模型全部参数的小部分，甚至个位数百分比，同时达到与fully fine-tuning相媲美的结果，导致任务特定参数减少10-1000倍。
  - Parameter-efficient fine-tuning，尤其是adapter-tuning和prefix-tuning，在高数据环境下（即监督样本的数量），可以获得与fully fine-tuning相媲美的性能，在低数据环境下性能更佳，并且对训练中未见过的主题示例有更好的外推能力。
  - Parameter-efficient fine-tuning在准确性上优于prompt engineering。此外，它比离散的prompt engineering更有表达性，后者需要匹配真实词的嵌入。
  - 尤其是prefix-tuning允许在推理时进行混合任务批处理（不同任务的查询可以在同一个批次中），因此在GPU共享方面具有计算效率（注：adapter-tuning的推理不能批处理）。
  - 一些研究显示，与adapter-tuning相比，prefix-tuning需要的参数数量显著减少，同时保持了可比较的性能。直觉是，prefix-tuning尽可能保持预训练的语言模型不变，因此比adapter-tuning更多地利用了语言模型。
- 劣势：
  - 在一些任务（如大数据量摘要）上可能不如 fully fine-tuning；
  - 往往需要比 prompt engineering 更多的监督样本。

## 适配器管理与部署（Serving）

- 适配器装载：推理时“热插拔” LoRA/Prefix，便于多任务共用底模；
- 合并与回写：极简部署可将 LoRA 合并回权重（牺牲灵活性换简化）；
- 批处理与并发：注意不同任务的前缀长度差异；结合 vLLM/paged‑KV 提升吞吐；
- 版本与评估：多任务多适配器时，按环境（dev/staging/prod）管理版本与 A/B 实验。
---

# Prompting

**Prompt-tuning**：与 prefix-tuning 相似，可视为简化版。冻结底模，仅为每个任务添加一组可学习的 soft tokens 并置于输入前，使其在少样本下压缩监督信号、缩小与 fully fine-tuning 的质量差距；易于在多任务共用一个底模时扩展。

**Instruction-tuning**：在一个以“指令”形式表达的任务集上对预训练模型做监督微调，使其更好地响应指令，并显著改善 zero-shot 表现。

2022年，Instruction fine-tuning因为该技术在不损害模型的泛化能力的情况下显著提升模型性能而获得了巨大的流行。通常，一个pre-trained LM会在一系列语言任务上进行fine-tuning，并在fine-tuning时未见过的另一组语言任务上进行评估，以证明其泛化能力和zero-shot能力。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm-practice3.png)

优劣对比：

- 优势：
  - 2022年在语言模型（LM）领域的一个突破是，instruction fine-tuning成为了提高大型语言模型（LLMs）整体性能和有用性的state-of-the-art。
  - 特别是在零样本学习场景中（即面对未见过的任务时），instruction fine-tuning显著提高了模型的性能，因此经过Instruction fine-tuning的LLMs比那些更多为特定用例而精细fine-tuning的模型具有更强的泛化能力。
- 劣势：
  - 这种方法更新了整个模型的参数，而不是在paramater-efficient fine-tuning中冻结其中一部分。这意味着它不会带来参数高效fine-tuning所具有的成本优势。然而，鉴于instruction fine-tuning产生的模型相比parameter-efficient fine-tuning更具有泛化能力，instruction fine-tuning的模型仍然可以作为服务于多个下游任务的通用模型。本质上，这取决于你是否拥有进行instruction fine-tuning所需的指令数据集以及训练预算。
  - Instruction fine-tuning在自然以指令形式表述的任务上普遍有效（e.g., NLI, QA, translation, struct-to-text），但对于直接以语言建模形式构建的任务（例如，推理）就有点更棘手了。为了使instruction fine-tuning在推理任务中也能比传统fine-tuning表现得更好，需要在instruction fine-tuning的监督样本中包含chain-of-thought示例。

**Reinforcement Learning from Human Feedback（RLHF）**：
RLHF 是instruction fine-tuning的扩展，在instruction fine-tuning步骤之后增加了更多步骤，以进一步纳入人类反馈。
预训练的大型语言模型（LLMs）经常表现出非预期的行为，如捏造事实（也称为幻觉）、生成有偏见或有害的回答，或简单地不遵循用户指令。这是因为许多最新的语言模型使用的语言建模目标——从互联网上的网页预测下一个词——与“有帮助且安全地遵循用户指令”的目标不同。
通过人类反馈进行强化学习是一种fine-tuning技术，帮助语言模型按照用户的明确意图（如遵循指令、保持真实）和期望行为（不表现出有偏见、有害或其他有害特征）行动。
这项技术的最佳示范是由OpenAI提供的。他们采用了他们的预训练GPT-3模型，使用RLHF进行了fine-tuning，并派生出他们称为InstructGPT的更加对齐的模型。此外，ChatGPT也是使用RLHF在更高级的GPT模型系列（称为GPT-3.5）上派生出来的。

RLHF步骤：
1. Instruction-tuning：收集期待模型完成的行为并以此通过监督学习fine-tuning LLMs；
2. 收集模型输出之间比较的数据集，其中标记器指示他们对于给定输入更喜欢哪个输出。 然后，训练奖励模型来预测人类偏好的输出。
3. 采用经过训练的奖励模型，并使用强化学习针对奖励模型优化策略（step 2、3不断循环）。

**RLHF 小结**：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm-practice4.png)

### DPO/IPO/RLAIF 简述

- DPO（Direct Preference Optimization）：直接用偏好对比数据优化策略，无需显式训练奖励模型，稳定、计算轻；
- IPO/KTO：将偏好优化建模为信息投影或温度调度，缓解过拟合与偏置；
- RLAIF：用强模型/专家系统合成偏好数据替代人类标注，快速冷启；
- 常见陷阱：奖励/偏好过优化（specification gaming）、风格单一化、拒答过度；需加入多样化偏好与安全对抗样本。

在实际应用中，常有 prompting 更高效的场景。通常有两条路径：

- 使用一个开源的self-trained LLM，设计或者调整prompts，然后自己进行推理；
- 使用一个商用的LLM API服务（例如GPT-4），然后在它们的调试平台设计prompts。

**Prompt engineering**：在 prompt 中组合任务描述、指令与示例。常基于已 instruction-tuned 的底模，冻结参数以便共享服务多任务。

Prompt的直觉是，有一个合适的上下文可以在不改变其参数的情况下引导LM。例如，如果我们想让LM生成一个词（e.g., Bonaparte），我们可以在前面添加它的常见搭配作为上下文（e.g., Napoleon），这样语言模型会给所需词汇分配更高的概率。

**Manual prompt design vs. Automated prompt design**：

- Manual prompt design：最自然而然的创建prompts的方法，就是通过人工的形式进行prompt设计。
- Automated prompt design：自动化prompt设计涉及自动搜索在离散空间中描述的prompt模板，通常映射到自然语言短语。

Prompt Engineering 优劣分析：

- 优势：
  - 少量样本prompt可以大大减少对特定任务数据的需求。在收集大量监督样本不可行的情况下，prompt engineering可以用很少的例子来工作。 
  - 它也适用于零样本设置（没有提供监督样本）。 
  - 从效率上讲，它不需要更新参数。 
  - 没有灾难性遗忘，因为LMs的参数保持不变。 
  - 通常在分布外泛化方面比大多数其他fine-tuning方法表现得更好。 
  - Prompt engineering生成人类可解释的prompt模板，因此在需要人类努力/主题专家参与迭代的地方更加直观。
- 劣势：
  - 因为prompt是提供任务规范的唯一方法，有时需要大量工程才能实现高准确度。然而，找到健壮的、一致的和可解释的prompt选择规则极其困难。
  - 提供许多受监督的样本在推理时可能会很慢，特别是如果你没有使用商业LM API之一，因此不能轻易地从大型任务特定样本中获益（如果可用的话）。
  - 通常与fine-tuning相比表现较差，特别是如果有大型任务特定样本可用，这取决于你正在prompt的base LM。
  - 每次模型进行推理时处理所有prompt的输入-目标对，可能会因高推理场景而产生显著的计算成本。
  - LLMs只能依赖于有限长度的上下文（例如，GPT-4的8k令牌），prompt不能充分利用比上下文窗口更长的训练集。

**Chain-of-thought（CoT）**：在 prompt 中提供中间推理步骤示范，以解锁模型的推理能力，常与 self-consistency（多样化采样投票）结合提升稳健性。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/llm-practice5.png)

直觉是，chain-of-thought提供了一个可解释的窗口，用于观察模型的行为，表明它可能是如何得出特定答案的，并提供了在推理路径出错处进行fine-tuning的机会。Chain-of-thought推理可用于数学文字题、常识推理和符号操作等任务。研究表明，在足够大的pre-trained LLM（数十亿参数）中，可以容易地引出chain-of-thought推理，但对于较小的LLM则效果不佳。

**Prompt chaining**：将整体任务拆解为沿 DAG 的子任务链，使用上一步输出作为下一步输入，以更透明可控地完成复杂任务。与 ReAct（推理 + 行动）模式、工具调用结合可进一步提升可解释性与成功率。

## Prompt 设计与防护实践

- 结构化输出：使用 JSON/函数调用或语法约束（Grammar/Regex），减少解析错误；
- Few‑shot 选择：基于语义相似度自动选择示例，避免随机样本带来噪声；
- 明确边界：在系统提示中声明拒答、安全与来源引用规则；
- 注入防护：分离系统/开发者/用户层级，使用“不可覆盖段”，加入“不要执行/不要浏览”等负面指令与模板清洗；
- 工具与检索：显式工具规范（输入/输出模式），对检索结果做“证据引用”。

有诸如 *W&B Prompts*这样的工具，提供了一个交互式界面，以视觉方式检查他们的prompt chain，以帮助prompt设计者进行LM链的创作过程。本质上，Prompts 允许用户审查其 LM 采取的步骤以提供输出，从而允许对过程进行细粒度的理解。除了其他优势，prompt chain使得调试 LLMs（大型语言模型）变得更加容易。

**LangChain**：还有一些复杂的LLM管道构建框架，比如LangChain。它协助构建和维护由引导LM调用（可以涉及对不同LM API的多次调用）、外部工具（网络搜索、计算器、字典等）以及LLM Agents组成的管道，这些LLM Agents可以将输入流量路由到最合适的动作以进行自动化规划和问题解决。设计基于LLM的管道和系统是提高系统整体性能的有效方式，前提是管道的部分设计用于缓解LLM的不可靠和非确定性输出，并赋予LLM使用外部可用的专门执行确定性任务的工具（例如计算器）。

随着商业 LLM（如 GPT 系列）性能提升，生态从“训练模型”转向“构建基于 LLM 的系统”，通过检索、工具与编排提升端到端效果。



**实践建议与评估**：

- 数据与标注：优先构建高质量、可复用的标注集；引入一致性/偏好数据以提升对齐质量。
- 评估体系：同时跟踪“内在指标”（困惑度、准确率、F1）与“外在指标”（延迟、吞吐、成本、拒答率）。
- 部署与成本：优先考虑量化（INT8/INT4）、KV 复用、批处理与连续批处理；用 LoRA/QLoRA 在共享底模上热插拔任务适配器以降低多任务成本。

### 一些可复用的超参与经验（参考）

- LoRA：r=16/32，α=32/64，dropout=0.05；target=[q,k,v,o,gate,up,down]；
- 学习率：PEFT 5e‑4 ~ 1e‑3；全量 2e‑5 ~ 5e‑5；warmup 3–5%；
- Batch：尽可能大（配合累积）；混合精度 bf16；梯度裁剪 1.0；
- 评估：每 N 步 checkpoint + 验证集；上线前小流量 A/B；
- 观测：内在（loss/acc/F1）+ 外在（延迟/吞吐/拒答/成本）双轨监控。

**小结**：未来仍会涌现新的基础模型，但大多数组织会更少地训练自有 GPT 变体，而更多地在现有模型上做 fine-tuning 或 prompt engineering。关键在于以工程化方式拆解问题、构建数据与评估闭环，并在成本边界内持续迭代。
