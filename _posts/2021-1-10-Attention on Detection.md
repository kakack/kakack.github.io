---

layout: post
categories: [Algorithm]
tags: [Detection, Deep Learning]

---


# Abstract

注意力机制在NLP领域和机器翻译领域已取得了巨大成就，因此在引入视觉领域时，也是颇受期待。注意力机制可以理解为模仿人类观察画面的方式，会主要关注一些局部重要的信息来增进其对整体画面信息的理解能力。整体而言可以概括为两个方面：

1. 注意力机制需要决定整体输入中哪部分需要被额外关注；
2. 从关键部分进行特征提取，以获得重要信息。

在NLP领域，对于语言信息的理解很大程度上会以来语句上下文语境和先后顺序。同样在视觉领域，对于图像信息理解也存在从整体到局部或从局部到整体的观察理解顺序，同样如车牌识别、手写信息识别等方面，也会依赖画面从左往右或者从上往下的顺序。而循环神经网络（Recurrent Neural Network，RNN）本身就特别适合处理这一类order ruled的信息，因此注意力机制往往会在RNN上应用。

但是在图像层面，输入神经网络的往往都是一整张图片，只有像素之间相互位置信息，并没有如NLP中语序先后顺序，而且从人类理解能力角度来看，并没有什么可以模式化或者规则化的观察手段，且观察顺序也会因人而异，因此很难用固定规则的形式来确定图片中信息理解顺序。因此在原有RNN基础上需要进行一定的改造。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-1.png)

# Attention on RNN

对于端到端的RNN来说，在NLP领域又被称为Sequence to Sequence（seq2seq），即输入一句话，输出另一句话，中间包括encoder和decoder两个部分。可以总结为：

1. 建立一个编码器encoder和一个解码器decoder的非线性模型，具有足够多的Neural Network Parameters可以存储足够多的信息；
2. 在关注句子整体信息时，每次翻译到下一个词语时，需要对不同词语赋予不同的权重weight，这样在再解码的时候可以同时考虑到整体和局部的信息。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-2.png)

# Different kinds of Attention

现在常见的注意力方法有两类：

1. 基于强化学习Reinforcement Learning的方法，通过收益函数Rewarding来反馈激励，让模型更加关注某个局部的细节；
2. 基于梯度下降Gradient Decent的方法，通过目标函数objective function和相应的优化函数optimization function来做。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-3.png)
---

# Reference 

- [Spatial Transformer Networks（STN）](https://arxiv.org/pdf/1506.02025.pdf)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
- 