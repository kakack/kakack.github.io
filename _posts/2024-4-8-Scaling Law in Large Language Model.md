---

layout: post
tags: [Engineering, LLM]
title: Scaling Law in Large Language Model
date: 2024-04-3
author: Kaka Chen
comments: true
toc: true
pinned: false

---
# Basic

**Scaling Law定义**：随着模型大小、数据集大小和用于训练的计算浮点数的增加，模型的性能会有规律性的提高。并且为了获得最佳性能，所有三个因素**必须同时放大**。当不受其他两个因素的制约时，模型性能与每个单独的因素都有**幂律关系（Power Law Relationship）**。因此，当这种幂率关系出现时，我们是可以提前对模型的性能进行预测的。

**Key Scaling Laws：**

- **模型尺寸：** LLM 的性能通常随着模型参数数量的增加而提升。
- **训练数据量：** 随着训练数据量的增加，LLM 的性能也会提高。
- **训练时间：** 训练时间越长，LLM 的性能通常越好。
- **训练硬件：** 训练硬件的性能（例如 GPU 数量和速度）也会影响 LLM 的性能。

**Scaling Law 的意义：**

Scaling Law 对于理解和改进 LLM 至关重要，因为它提供了以下见解：

- **模型设计：** Scaling Law 指导模型架构师选择最佳模型尺寸和参数数量以实现特定任务。
- **训练策略：** Scaling Law 帮助研究人员确定最佳训练数据量和时间以最大化模型性能。
- **硬件选择：** Scaling Law 告知研究人员和从业人员所需的训练硬件类型和规模以实现所需性能水平。

1. 类似GPT4结构的模型估算方法，有浮点运算量（FLOPs）$C$、模型参数$N$以及训练的tokens数$D$之间的关系：$C \sim 6ND$

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240409-1.jpg)

1. 模型的最终性能主要与浮点运算量（FLOPs）$C$、模型参数$N$以及训练的tokens数（训练数据量）$D$三者相关，而与模型的具体结构(层数/深度/宽度)基本无关。
2. 对于计算量（FLOPs）$C$、模型参数$N$以及数据大小$D$，当不受其他两个因素制约时，模型性能与每个因素都呈现**幂律关系。**
3. 为了提升模型性能，模型参数$N$以及数据大小$D$需要同步放大，但分别放大的比例还存在争议。
4. Scaling Law适用于语言模型以及其他模态以及跨模态的任务。

**核心公式**：

$$
L(x)=L_{\infty}+(\frac{x_0}{x})^\alpha \\ \begin{align} L_\infty \approx S({True}) &&& \text{"Irreducible Loss"} \\ (\frac{x_0}{x})^{\alpha_x} \approx D_{KL}({True}||{Model}) &&& \text{"Reducible Loss"}  \end{align} \\ 
$$

- 第一项$L_\infty$是指无法通过增加模型规模来减少的损失，可以认为是数据自身的熵（例如数据中的噪音）；
- 第二项$(\frac{x_0}{x})^\alpha$是指能通过增加计算量来减少的损失，可以认为是模型拟合的分布与实际分布之间的差。

根据公式，增大$x$(例如计算量$C$)，模型整体loss下降，模型性能提升；伴随$x$趋向于无穷大，模型能拟合数据的真实分布，让第二项逼近0，整体趋向于$L_\infty$。

其中以GPT4、Baichuan2、MindLLM为代表的LLM均在论文中指出符合这一scaling law。

# Implement

根据幂律定律，模型的参数固定，无限堆数据并不能无限提升模型的性能，模型最终性能会慢慢趋向一个固定的值。按照原有思路可以进行scaling law实操。

首先准备充足的数据（例如1T），设计不同模型参数量的小模型(例如0.001B - 1B)，独立训练每个模型，每个模型都训练到基本收敛（假设数据量充足）。根据训练中不同模型的参数和数据量的组合，收集计算量与模型性能的关系。然后可以进一步获得**计算效率最优**时，即同样计算量下性能最好的模型规模和数据大小的组合，模型大小与计算量的关系，以及数据大小与计算量的关系。