---

layout: post
categories: [Deep Learning]
tags: [Deep Learning， Neural Network，Object Detection]

---

# 前言

换工作后，开始有了真正的实际应用业务量和比较系统而聚焦的研究调研方向，过去两周主要针对基于卷积神经网络和其衍生网络，本文主旨在于简单介绍一下这一整套update pipeline：`Selective Search, Spartial Pyramid Pooling，R-CNN，Fast R-CNN，Faster R-CNN`，以供之后参考，暂不涉及模型的具体调优选参细节。



---

# Selective Search

[
Selective Search for Object Recognition 
Jasper R. R. Uijlings, Koen E. A. van de Sande, Theo Gevers, Arnold W. M. Smeulders](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib)

总体而言是用了segmentation和grouping的方法来进行object proposal，然后用了一个SVM进行物体识别。

需求来源是，在物体识别时候，往往需要根据多种特征模式来进行物体层级之间的区分以找出独立存在的各个物体，区分依据诸如颜色、纹理、嵌套结构等。因此SS采取了Multiscale的办法，即将原图分割成多个region，再由多个region grouping成更大的region，重复该过程，直到最后生成multiscale的region了。

## 流程概述：

- 输入：彩色图片（三通道）
- 输出：物体位置的可能结果L

![](http://ope2etmx1.bkt.clouddn.com/2018822-1.jpg)

- 使用[Efficient GraphBased Image Segmentation](http://cs.brown.edu/people/pfelzens/segment/)中的方法来得到初始化region
- 得到所有region之间两两的相似度
- 合并最像的两个region
- 重新计算新合并region与其他region的相似度
- 重复上述过程直到整张图片都聚合成一个大的region
- 使用一种随机的计分方式给每个region打分，按照分数进行ranking，取出top k的子集，获取每个区域的Bounding Boxes，就是selective search的结果

其中涉及到的相似度有：颜色相似度、纹理相似度、大小相似度、吻合相似度

## Object Recognition

![](http://ope2etmx1.bkt.clouddn.com/2018822-2.png)

特征提取+SVM

- 特征用了HoG和BoW
- SVM用的是SVM with a histogram intersection kernel
- 训练时候：正样本：groundtruth，负样本，seletive search出来的region中overlap在20%-50%的。
- 迭代训练：一次训练结束后，选择分类时的false positive放入了负样本中，再次训练

---

# IOU

IOU，aka Intersection-over-Union

---

# SSP

---

# R-CNN

---

# Fast RCNN

---

# Faster RCNN

---

# Conclusion




