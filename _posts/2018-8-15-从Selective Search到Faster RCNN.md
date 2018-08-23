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

- 使用[Efficient GraphBased Image Segmentation](http://cs.brown.edu/people/pfelzens/segment/)中的方法来得到初始化region（Kinds of greedy method）
- 得到所有region之间两两的相似度
- 合并最像的两个region
- 重新计算新合并region与其他region的相似度
- 重复上述过程直到整张图片都聚合成一个大的region
- 使用一种随机的计分方式给每个region打分，按照分数进行ranking，取出top k的子集，获取每个区域的Bounding Boxes，就是selective search的结果

其中涉及到的相似度有：颜色相似度、纹理相似度、大小相似度、吻合相似度、综合各种距离

## Object Recognition

![](http://ope2etmx1.bkt.clouddn.com/2018822-2.png)

特征提取+SVM

- 特征用了HoG和BoW
- SVM用的是SVM with a histogram intersection kernel
- 训练时候：正样本：groundtruth，负样本，seletive search出来的region中overlap在20%-50%的。
- 迭代训练：一次训练结束后，选择分类时的false positive放入了负样本中，再次训练

---

# IOU

IOU，aka ***Intersection-over-Union***，简单来讲就是模型产生的目标窗口和原来标记窗口的交叠率。具体我们可以简单的理解为： 即检测结果(DetectionResult)与 Ground Truth 的交集比上它们的并集，即为检测的准确率 IoU :

<center><img src="http://latex.codecogs.com/gif.latex?IOU=\frac{DetectionResult&space;\bigcap&space;GroundTruth}{DetectionResult&space;\bigcup&space;GroundTruth}"/></center>

或者

<center><img src="http://latex.codecogs.com/gif.latex?IOU=\frac{AreaOfOverlap}{AreaOfUnion}"/></center>

其中所谓的ground truth即正确的标注物体位置。

IoU是一个简单的测量标准，只要是在输出中得出一个预测范围(bounding boxex)的任务都可以用IoU来进行测量。为了可以使IoU用于测量任意大小形状的物体检测，我们需要：
 
1. ground-truth bounding boxes（人为在训练集图像中标出要检测物体的大概范围）； 

2. 我们的算法得出的结果范围。

也就是说，这个标准用于测量真实和预测之间的相关度，相关度越高，该值越高。

---

# SSP

[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)

SSP，aka ***Spatial Pyramid Pooling***，需要解决的问题有二：

1. 现有的CNN做图像识别要求输入图像尺寸固定，因此在喂入数据时会对图像做crop或wrap等操作，丢失一定图像信息。
2. 一个CNN通常分为卷基层conv和全连接层fc两部分，所以事实上单张图片只需要计算一遍卷积特征就可以，而将所有region proposals都做缩放会产生大量计算。

根据这两个原则，使用`Spatial Pyramid Pooling Layer`层代替掉原先卷积层中最后一层Pooling层，使得任意尺寸的卷积层输出经过SSP层之后，都输出固定大小的向量。这个SSP利用的是一种SPM（Statistical Parametric Mapping）的思想，即将一幅图分解成若干个份，如1份、4份、8份，然后将其每一块提取特征后融合在一起，就能兼容多个尺度的特征了。而在SSP中，将份数固定为自然数的平方次份，如1、4、9、16等，其数学依据在于任何一个自然数都可以写成若干个自然数的平方和。

![](http://ope2etmx1.bkt.clouddn.com/2018822-3.jpg)

SSP的优点：

- 不论输入尺寸如何，都可以输出固定尺寸
- 使用多个窗口Pooling Window
- 可以用同一个图片的不同scale作为输入
- 降低Overfitting，更容易收敛，可自检


---

# R-CNN

[Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation
R. Girshick, J. Donahue, T. Darrell, J. Malik](https://dl.dropboxusercontent.com/s/293tu0hh9ww08co/r-cnn-cvpr.pdf?dl=0)

RCNN所解决的两个问题：

1. 传统目标识别算法以sliding windows的办法，而RCNN采取Selective Search预先生成region proposal，之后只处理这些proposals
2. 传统目标识别需要在区域中人工设定特征（Haar， HOG），而RCNN则使用深度神经网络进行特征提取。

整体框架：（其组成部分包括生成类独立的region proposal模块、提取特征的CNN模块和一组SVM分类器）

1. 用selective search方法划分2k-45k个region；
2. 分别对每个region提取特征，最后将提取到的特征送到k(k的取值与类别数相等)个svm分类器中识别以及送到一个回归器中去调节检测框的位置；
3. 将k个SVM分类器中得分最高的类作为分类结果，将所有得分都不高的region作为背景；
4. 通过回归器调整之后的结果即为检测到的位置。

![](http://ope2etmx1.bkt.clouddn.com/2018822-4.png)


其中关键步骤有：

- **候选区域**：使用selective search：过分分割->穷举依靠相似度合并->输出最大IOU的region proposal
- **特征提取**：通过训练好的Alex-Net，先将每个region固定到227x227的尺寸，然后对于每个region都提取一个4096维的特征。在resize到227x227的过程中，在region外面扩大了一个16像素宽度的边框，region的范围就相应扩大了，考虑了更多的背景信息。
- **CNN训练**：首先拿到Alex-Net在imagenet上训练的CNN最为预训练，然后将网络的最后一个全连接层的1000类改为N+1（N为类别的数量，1是背景）来微调同于提取特征的CNN。为了保证训练只是对网络的微调而不是大幅度变化，调整学习率为0.001。计算每个region proposals与ground-truth的IoU，IoU阈值设为0.5，大于0.5的IoU作为正样本，小于0.5的为负样本。在训练的每一次迭代中都使用32个正样本（包括所有类别）和96个背景样本组成的128张图片的batch进行训练（这么做是因为正样本图片太少了）。
- **SVM分类器训练**：训练N（N为类别数）个线性svm分类器，分别对每一个类做一个二分类。在这里，作者将IoU大于0.5的作为正样本，小于0.3的作为负样本。（为什么不是0.5：因为当设为0.5的时候，MAP下降5%，设为0的时候，MAP下降4%，所以取中间值0.3）。后面说到softmax在VOC 2007上测试的MAP比SVM的MAP低，因为softmax的负样本（背景样本）是随机选择的即在整个网络中是共享的，而SVM的负样本是相互独立的，每个类别都分别有自己的负样本，SVM的负样本更“hard”，所以SVM的分类准确率更高。
- **Bounding Box Regression**：通过训练一个回归器来对region的范围进行一个调整，因为region最开始只是用selective search的方法粗略得到的，通过调整以后可以得到更加精确的位置。


---

# Fast RCNN

Fast R-CNN主要解决R-CNN的以下问题：

1. 训练、测试时速度慢: R-CNN的一张图像内候选框之间存在大量重叠，提取特征操作冗余。而Fast R-CNN将整张图像归一化后直接送入深度网络，紧接着送入从这幅图像上提取出的候选区域。这些候选区域的前几层特征不需要再重复计算。
2. 训练所需空间大: R-CNN中独立的分类器和回归器需要大量特征作为训练样本。Fast R-CNN把类别判断和位置精调统一用深度网络实现，不再需要额外存储。

![](http://ope2etmx1.bkt.clouddn.com/2018822-5.png)


---

# Faster RCNN

![](http://ope2etmx1.bkt.clouddn.com/2018822-6.png)


---

# Conclusion




