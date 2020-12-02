---

layout: post
categories: [Computer Vision]
tags: [Detection, Deep Learning]

---

# Abstract

Anchor Box最早是在Faster-R-CNN中被提到使用，此后在SSD、YOLOv2、YOLOv3等一系列目标检测网络中被普遍使用。Anchor Boxes实质上是一组人为预先设定的检测尺寸框，各个anchor box之间都有各自不同的尺寸和长宽比，以适应不同的被检物体类别。

`
For each sliding window, the networks make multiple simultaneous proposals, where is the number of proposals for each sliding window location is k. The reg layer gives 4k outputs containing the coordinates of the k boxes and the cls layer had 2k scores that estimate the probability of being an object or not object for each proposal. These k proposals are estimated from k reference boxes, which are called anchors. An anchor for a given sliding window, has a scale and aspect ratio associated with it, in their paper Ren et. al have used 3 scales and 3 aspect ratios, this yields 9 anchors at each sliding window location. For a convolutional feature map of size w×h the total number of anchors = w×h×k.
`

---

# Why use anchor boxes?

在anchor boxes之前，我们常用的物体检测方法有两种：

### 1，滑动窗口 Sliding Windows

简单而粗糙的遍历轮询方法，使用固定尺寸的窗口，在feature map上每一次移动固定步长，从左往右、从上往下，逐个遍历完整个feature map，把被窗口盖住的内容输入到后续的卷积神经网络里进行计算，得到分类标签和位置信息等。但是滑动窗口的缺陷也非常明显：1，由于窗口尺寸固定，步长固定，因此不适合形变较大的物体；2，窗口总量较多，所以需要的运算量较大。

### 2，区域建议 Region Proposal

这是R-CNN系列中引入的方法，如Faster R-CNN模型中分别使用CNN和RPN(Regional Proposal)两个网络。其中区域建议网络不负责图像的分类，它只负责选取出图像中可能属于数据集其中一类的候选区域。接下来就是把RPN产生的候选区域输入到分类网络中进行最终的分类。其中Selective Search是更早的一种获得RP的方法，主要依赖的是一些关于图像的先验知识，如颜色、纹理等。可以分为四个步骤：1，将分割的图片画出多个框，把所有框放入列表Region中；2，根据相似程度（颜色，纹理，大小，形状等），计算Region中框之间的俩俩形似度，把相似度放入列表A中；3，从列表A中找出相似度最大的俩个框a,b并且合并；4，把合并的框加入列表Region中，从A中删除和a，b相关的相似度，重复步骤2，直至清空A。

但是无论哪种方法，都无法很好解决两个问题：1，一个窗口只能检测一个目标；2，无法检测尺寸很大或者长宽比很极端的物体。

![](https://github.com/kakack/kakack.github.io/blob/master/_images/20201202-1.jpg?raw=true)

如上图作为输入，如果我们用3x3网格，而需要被检测的人和汽车的中心恰好都落在同一个网格中。假设我们用y作为这个格子的输出向量，y=[p_c, b_x, b_y, b_h, b_w, c_1, c_2, c_3]^T，其中p_c表示检测置信度，[b_x, b_y, b_h, b_w]表示物体bounding box位置（左上角x、y和高h、宽w），[c_1, c_2, c_3]是以one-hot编码形式表示的三种类别（如人、汽车、摩托车）中的一种。因此从一个输出向量y中我们只能选择一种类型作为检测输出。

那么anchor box的尺寸该怎么选择？目前anchor box的选择主要有三种方式：

- 人为经验选取
- k-means聚类
- 作为超参数进行学习


