---

layout: post
tags: [Detection, Deep Learning, Computer Vision]
title: Attention on Detection
date: 2021-01-10
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

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

# Look Closer to See Better

这是一篇非常经典的attention on cv的文章，工作的目的是给图像中的鸟类进行分类，包括种类识别和属性识别等内容。既然是针对图片中的鸟类，那么算法需要着重关注的局部信息自然集中在针对鸟的像素上，包括鸟的各个身体部分、整体形态、颜色等信息。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-4.png)

本文提出了一个基于CNN的attention mechanism，称为recurrent attention convolutional neural network（RA-CNN），该模型递归地分析局部信息，从局部的信息中提取必要的特征。同时，在 RA-CNN 中的子网络（sub-network）中存在分类结构，也就是说从不同区域的图片里面，都能够得到一个对鸟类种类划分的概率。这种从局部得到信息的方法也称作Attention Proposal Sub-Network（APN）。这个 APN 结构是从整个图片（full-image）出发，迭代式地生成子区域，并且对这些子区域进行必要的预测，并将子区域所得到的预测结果进行必要的整合，从而得到整张图片的分类预测概率。

图像整体输入network后，首先经过第一个conv blob会得到一个分类概率，同时输出一个坐标值表示子图的中心点位置和子图的scale size，然后将对应表示的子图输入下一个conv blob得到新的分类概率和子图位置尺寸，以此不断迭代得到越来越细致的子图以聚焦到核心关键区域，最终将所有分类概率整合得到整图的识别结果。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-5.png)

由此可见，其中的关键点在于：

1. 分类概率的计算，也就是最终loss function的设计；
2. 从上一张图到下一张图的坐标和尺寸大小缩放计算方法。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-6.png)


# Multiple Granularity Descriptors for Fine-grained Categorization

本文的工作同样是鸟类分类，不过使用了更贴近生物学的分层分类方法，通常把鸟类分作科、属、种三类，对应三个不同层级的网络。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-7.png)

从前往后的顺序是检测网络（Detection Network），区域发现（Region Discovery），描述网络（Description Network）。并行的结构是 Family-grained CNN + Family-grained Descriptor，Genus-grained CNN + Genus-grained Descriptor，Species-grained CNN + Species-grained Descriptor。而在区域发现的地方，作者使用了 energy 的思想，让神经网络分别聚焦在图片中的不同部分，最终的到鸟类的预测结果。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-8.png)

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-9.png)

# Recurrent Models of Visual Attention

这是最早利用强化学习方式在cv中引入attention的手段之一，使用收益函数来进行模型训练，从整体和局部两方面提取到必要信息。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-10.png)

整体而言其结构是RNN，每一个阶段都会将得到的信息和坐标传递给下一阶段，但是与RA-CNN不同的是，只会在最后一步对分类概率进行计算。这本质上还是将整图以某种时间序列分步输入网络，一次只处理整图一部分信息，并在处理过程中计算出接下去需要处理的位置和之前已经处理完的任务内容。

# Multiple Object Recognition with Visual Attention

与上文中的RMVA不同的是，这个模型提供了两层RNN，并在最上层输入原始整图。其中enc是编码网络encoder，$r_i^{(1)}$是解码网络decoder，$r_i^{(2)}$是注意力网络，输出概率在解码网络的最后一个unit输出。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-11.png)

在门牌识别里面，该网络是按照从左到右的顺序来进行图片扫描的，这与人类识别物品的方式极其相似。除了门牌识别之外，该论文也对手写字体进行了识别，同样取得了不错的效果。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-12.png)

# Squeeze-and-Excitation Networks（SENet）

SENet是一个非常通用的模块化结构，可以嵌入当今各种主流网络。其主旨是基于channel的注意力机制。根据输入图像池化得到feature map后，对每个其中每一个channel赋予一个权重值，将当前channel值和对应权重值的乘积作为真正参与计算的feature map，以此区分各个channel的重要性，针对其中重要性较高的channel进行更多的信息关注。如下图所示，不同重要性channel在最终的feature map中用不同颜色的块表示：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-13.png)

SENet block处理的过程可以理解为两部分：Squeeze和Excitation。其中$c*1*1$的global pooling称之为squeeze，而之后的两个fully connect layer称为excitation，最终用sigmod将输出限制在$[0, 1]$范围内，把这个值作为scale权重乘到各个channel上，作为下一级的输入。这么做的原理就是通过控制权重scale的大小把重要channel的特征增强，无关channel的特征削弱。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-14.png)

# Convolutional Block Attention Module（CBAM）

CBAM相对于SENet的区别就在于其即关注不同channel的重要性，也关注某个channel中不同位置pixel的重要性。Convolutional Block Attention Module (CBAM) 表示卷积模块的注意力机制模块。是一种结合了空间（spatial）和通道（channel）的注意力机制模块。相比于senet只关注通道（channel）的注意力机制可以取得更好的效果。它相对于SE多了一个空间attension，这个空间其实就是宽高对应的方形或者说是一个通道对应的feature map，SE只关注通道，它既关注通道，也关注宽高。

基于传统vgg结构的CBAM模块：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-15.png)

基于如Resnet等shortcut结构的CBAM模块：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-17.png)

与SENet不同的是，一开始的pooling分为MaxPool和AvgPool两条支线，再将两个输出结果相加后做sigmod控制在$[0, 1]$范围内，最后跟feature map相乘。两种不同的polling意味着提取的高层次特征更加丰富，这是通道上的attention。

之后还会进一步做空间上的attention。首先将基于channel attention的feature结果再做一次MaxPool和AvgPool，但这次是在channel这个维度进行，即把所有输入channel全部pooling到2个实数，由$(h * w * c)$的形状transfer到两个$(h * w * 1)$的feature map，紧接着使用一个7*7的conv kernel形成新的$(h * w * 1)$的feature map。最后也是相同的scale操作，注意力模块特征与得到的新特征图相乘得到经过双重注意力调整的特征图。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/2021010-16.png)

---

# Reference 

- [Spatial Transformer Networks（STN）](https://arxiv.org/pdf/1506.02025.pdf)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
- [Look Closer to See Better：Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf)
- [Multiple Granularity Descriptors for Fine-grained Categorization](https://openaccess.thecvf.com/content_iccv_2015/papers/Wang_Multiple_Granularity_Descriptors_ICCV_2015_paper.pdf)
- [Recurrent Models of Visual Attention](https://arxiv.org/pdf/1406.6247.pdf)
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf)