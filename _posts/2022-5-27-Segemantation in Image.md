---
layout: post
categories: [Computer Vision]
tags: [Detection, Deep Learning]


---

# Brief

图像分割需要做的事情就是把一张图中的像素根据其所表示的不同内容而进行区分并打上标签，本质上可以看成一种聚类操作。图像分割方法主要可分为两种类型：语义分割和实例分割。语义分割会使用相同的类标签标注同一类目标（Segmantic Segmentation），而在实例分割中，相似的目标也会使用不同标签进行标注（Instance Segmentation）。深度学习之前的图像分割方法主要有如阈值化、基于直方图的方法、区域划分、k-均值聚类、分水岭，到更先进的算法，如活动轮廓、基于Graph的分割、马尔可夫随机场和稀疏方法。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-1.jpeg)



基于深度学习的图像分割模型的基本架构包括编码器与解码器。编码器通过卷积核提取图像特征，通过不断下采样将大尺寸低通道的输入图像变为小尺寸高通道的特征向量。解码器负责输出包含物体轮廓的分割蒙版，将特征向量恢复成大尺寸低通道的形状。此外，模型常见的特性还有skip连接、多尺度分析，以及最近使用的dilated卷积。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-2.jpeg)

# FCN（Fully Convolutional Network）

这是将深度学习应用于图像分割中的一大基石，证明了可以在可变大小的图像上以端到端的方式训练深层网络进行语义分割。FCN模型具有非常可观的有效性和普遍性，然而却存在推理性能较慢，无法有效考虑全局上下文信息，也不容易转换为3D图像。



![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-5.jpeg)

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-6.jpeg)

# RCNN-Based Model

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-4.jpeg)

# U-Net

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220527-3.jpeg)

以Unet为代表的一系列Encoder-Decoder Based Models同常见的CNN相比，会去掉最后一些将卷积输出大小降低的层，而将卷积恢复到输入样本的尺寸（$n_H \times n_W$），这就会用到Transpose Conv。常规的卷积会将一个较大尺寸的dimension缩小成一个较小尺寸的dimension，而transpose conv要做的事情恰好相反。通常的卷积操作是用filter在原input矩阵上进行卷积求和操作，而TC恰好与之相反，是将filter覆盖在output上反向推得到input。这是一种高效的dimension expand方法，同时又兼顾了上下文语义。

在Unet的前半部分，类似于normal conv，会采用一些max pooling来减小宽高，而channel会增加（红色箭头）。后半部分会采用trans conv不断扩大dimension，减小channel数（绿色箭头），同时从左到右会有一些skipped connection类似residual block作为输入进入每一层激活函数（灰色箭头）。



# Conclude

