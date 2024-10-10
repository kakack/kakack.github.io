---

layout: post
tags: [LLM, NLP]
title: The Needle In a Haystack Test
date: 2024-05-16
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

# 什么是大海捞针实验？

所谓的大海捞针实验（The “Needle In a Haystack” test）是设计用来评估LLM RAG系统在不同长短的上下文中的表现。它的工作原理是将特定的、有针对性的信息“针”（“Needle”）嵌入更大、更复杂的文本主体“草垛”（“Haystack”），目标是评估LLMs在大量数据中识别和利用这一特定信息的能力。

通常在RAG系统中，上下文窗口往往是信息溢出的状态，从矢量数据库返回的大量上下文与语言模型、模板和prompt中可能存在的其他内容的指令混杂在一起。大海捞针测试可以评估测试LLMs在这一混乱中找出具体细节的能力。

# 大海捞针实验的要点是什么？

- 不是所有的LLMs都是相同的，模型可以被以不同的目的和要求来训练得到；
- Prompt的细微差别都可能导致模型间的结果大不相同，一些LLMs需要更有针对性的prompt才能在特定的任务中表现出色；
- 在LLMs基础上构建模型时（尤其是当这些模型与私有数据相连时），有必要在整个开发和部署过程中评估检索和模型性能，看似微不足道的差异可能会导致性能的巨大差异。

# 如何创建大海捞针实验？

最早的大海捞针实验是被用在评测ChatGPT-4和Claude2.1这两个模型的召回率。其中一段这样的声明被放置在Paul Graham的文章中不同长度的片段的不同深度内：“The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day”，例如：

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240516-1.png)

然后模型会仅根据当前这段上下文内容，来回答“what the best thing to do in San Francisco was”这个问题。然后针对文章的0%（文档顶部）和 100%（文档底部）之间的不同深度中每1K个tokens代表的内容或每个模型的tokens数限制之间的不同上下文长度（GPT-4 为 128k，Claude 2.1 为 200k）重复此操作。下图记录了这两个模型的性能：

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240516-2.png)

正如所见，ChatGPT 的性能在少于64k个tokens时开始下降，而在100k及以上时急剧下降。有趣的是，如果“针”（needle）位于整段上下文的开头，模型往往会忽略或“忘记”它，而如果它位于结尾或作为第一个句子，模型的性能仍然稳定。

![img](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240516-3.png)

至于Claude，最初的测试并不顺利，最终检索准确率仅为27%。我们也观察到了类似的现象，随着上下文长度的增加，性能会下降，当针头隐藏在文档底部附近时，性能通常会提高，如果针头是上下文的第一句话，则检索准确率为100%。

---

# Reference

- [Long context prompting for Claude 2.1](https://www.anthropic.com/news/claude-2-1-prompting)