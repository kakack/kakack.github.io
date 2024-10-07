---
layout: post
tags: [Detection, Deep Learning, Computer Vision]
title: Segmentation in Image
date: 2022-05-27
author: Kyrie Chen
comments: true
toc: true
pinned: false



---

# Brief

图像分割需要做的事情就是把一张图中的像素根据其所表示的不同内容而进行区分并打上标签，本质上可以看成一种聚类操作。图像分割方法主要可分为两种类型：语义分割和实例分割。语义分割会使用相同的类标签标注同一类目标（Segmantic Segmentation），而在实例分割中，相似的目标也会使用不同标签进行标注（Instance Segmentation）。深度学习之前的图像分割方法主要有如阈值化、基于直方图的方法、区域划分、k-均值聚类、分水岭，到更先进的算法，如活动轮廓、基于Graph的分割、马尔可夫随机场和稀疏方法。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-1.jpeg)



基于深度学习的图像分割模型的基本架构包括编码器与解码器。编码器通过卷积核提取图像特征，通过不断下采样将大尺寸低通道的输入图像变为小尺寸高通道的特征向量。解码器负责输出包含物体轮廓的分割蒙版，将特征向量恢复成大尺寸低通道的形状。此外，模型常见的特性还有skip连接、多尺度分析，以及最近使用的dilated卷积。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-2.jpeg)

# FCN（Fully Convolutional Network）

这是将深度学习应用于图像分割中的一大基石，证明了可以在可变大小的图像上以端到端的方式训练深层网络进行语义分割。FCN模型具有非常可观的有效性和普遍性，然而却存在推理性能较慢，无法有效考虑全局上下文信息，也不容易转换为3D图像。FCN的核心贡献在于：

- **全卷积（convolutional）：**采样端对端的卷积网络，将普通分类网络的**全连接层换上对应的卷积层（FCN）**
- **上采样(upsample)**：即反卷积（deconvolution），恢复图片的位置信息等，反卷积层可以通过最小化误差学习得到。
- **跳跃连接(skip layer)**：通过连接不同卷积层的输出到反卷积层，来改善上采样很粗糙的问题。

相比于普通CNN，FCN允许整张原图直接作为输入，允许图片大小不固定。同时，FCN在最后几层做了修改以适用于语义分割应用中。它丢掉了CNN分类网络后端的FC层，进而保留了图片上的区域位置信息，又在其后加上了几层CNN来进一步分析处理，整合主干网络输出特征，最终它生成出有着C+1（C为目标类别数，+1是为了考虑进去图片背景）个channels的heat map（本质上可以理解为是cnn所产生的feature map）来。



![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-5.jpeg)

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-6.jpeg)

# RCNN-Based Model



RCNN以及其一系列衍生模型（Fast-RCNN、Faster-RCNN）将region proposal和CNN结合，是最初也是最经典的将深度学习在目标检测领域的应用。也是Two-Stage目标检测方法的标杆。其核心思想就是将目标检测任务分成两个部分：

1. 获得输入图像的region proposal。从最初在RCNN上应用selective search到Faster-RCNN上利用集成的RPN（Region Proposal Network），都是用于得到图像中具有相近特征的区域，也就是我们需要做的图像分割的工作；
2. 后续的CNN分类器将不同的region proposal分别分类并输出label。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-4.jpeg)

# U-Net

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-3.jpeg)

以Unet为代表的一系列Encoder-Decoder Based Models同常见的CNN相比，会去掉最后一些将卷积输出大小降低的层，而将卷积恢复到输入样本的尺寸（$n_H \times n_W$），这就会用到Transpose Conv。常规的卷积会将一个较大尺寸的dimension缩小成一个较小尺寸的dimension，而transpose conv要做的事情恰好相反。通常的卷积操作是用filter在原input矩阵上进行卷积求和操作，而TC恰好与之相反，是将filter覆盖在output上反向推得到input。这是一种高效的dimension expand方法，同时又兼顾了上下文语义。

在Unet的前半部分，类似于normal conv，会采用一些max pooling来减小宽高，而channel会增加（红色箭头）。后半部分会采用trans conv不断扩大dimension，减小channel数（绿色箭头），同时从左到右会有一些skipped connection类似residual block作为输入进入每一层激活函数（灰色箭头）。

# Conclude

如果只关注图像分割，我个人实践中尝试过FCN和UNet两种模型，其核心思想还是有很多类似之处，比如对于反卷积上采样和跳跃连接的应用。但是现在单纯以分割为目的的结构和应用其实是非常少的，因为首先这个领域整体发展就非常缓慢，几乎已经陷入瓶颈期，而且消耗算力较大，而我们工程上往往采取end-to-end的结构，把分割过程融入于目标识别的过程当中。但是其中一些特殊难点方面如小尺寸目标分割、轻量级语义分割、点云分割、实时分割等，仍具有一定挖掘价值。

# Reference

- [RCNN系列详解](https://blog.51cto.com/u_13977270/3397361)
- [2020入坑图像分割，我该从哪儿入手？](https://zhuanlan.zhihu.com/p/145009250)
- [语义分割系列——FCN](https://perper.site/2019/02/20/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E7%B3%BB%E5%88%97-FCN%E8%AF%A6%E8%A7%A3/)
- [基于深度学习的图像分割综述](https://zhuanlan.zhihu.com/p/141352661)

