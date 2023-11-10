---

layout: post
tags: [Detection, Deep Learning]
title: Cornernet and Centernet on Detection
date: 2022-01-04
author: Kaka Chen
comments: true
toc: true
pinned: false


---

# Why Anchor Free?

随着anchor在检测算法中的应用，不管是one or two stage的检测模型，都会在图片上放置密密麻麻尺寸不一的anchors，用来检测全图各个角落大小不一的目标物体。但是anchor based model有两个不足之处：

- Anchor amount过大，导致计算复杂度过高，对于绝大部分情况，只有其中很小一部分anchor能成功匹配到ground truth，而大量anchor作为负样本被丢弃。

- 整体引入大量hyper-parameters，以及针对anchor得到的bounding box所做的NMS操作，都会影响模型性能。

---

# CornerNet

CornerNet的文章取名为`Detecting Objects as Paired Keypoints`，言下之意就是用一对关键点来确定目标位置。我们主要从以下几点来描述：

1. 如何用anchor-free的方法表示目标；
2. 什么是corner pooling；
3. CornerNet的网络结构和损失函数。

## Locate the Object

在CornerNet中，我们预测的是目标bounding box的左上top-left和右下bottom-right两个角点作为识别关键点。网络在进行预测的时候，会为每个点分配一个embedding vector，属于同一物体的点的vector的距离较小。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-1.png)

作者选择corner而不是center的原因在于，一个中心点需要四个边决定，而一个角点只需要两个边，同时通过corner表示box相当于一种降维。

## Corner Pooling

如何预测角点的heatmap，就引入了corner pooling的思想。我们先生成整张图的heatmap，对于top-left而言，它首先包含两张相同的特征图，在每个像素点位置，它将第一个特征图右边的所有特征向量和第二个特征图正下方的所有特征向量都做max pooling，然后将pooling得到的两个结果相加。（可以用逆向最大值作为单方向corner pooling的值直接使用）

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-2.png)
## CornerNet Detail

### Network Architecture

CornerNet的网络结构主要分为以下几个部分

1. Backbone: Hourglass Network；
2. Head: 二分支输出 Top-left corners 和 Bottom-right corners，每个分支包含了各自的corner pooling以及三分支输出。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-3.png)

CornerNet借用了hourglass network作为他的backbone特征提取网络，这个hourglass network通常被用在姿态估计任务中，是一种呈沙漏状的downsampling 和 upsampling组合，为两个沙漏模块（hourglass module）头尾相连的效果。

在原有hourglass的基础上，作者做了以下的改进：

1. 在输入hourglass module之前，需要将图片分辨率降低为原来的1/4倍。本文采用了一个stride=2的7x7卷积和一个stride=2的残差单元进行图片分辨率降低。
2. 使用stride=2的卷积层代替max pooling进行downsample
3. 共进行5次downsample ,这5次downsample后的特征图通道为[256,384,384,384,512]
4. 采用最近邻插值的上采样（upsample),后面接两个残差单元

### Loss Function

#### Focus Loss

首先是detection中角点与目标ground truth之间的loss。输出的角点的heatmap有$C$个channel，$\tilde{M} \in R^{W/R\times H/R \times C}$，在这里$C$即数据集的种类（不设置background的类别）。每个channel都是0-1之间的数（可以看作一个概率），比如我们有$p_{cij}=1$就表示$(i, j)$位置的角点是类别$c$的角点的概率为1。$y_{cij}$表示该位置的ground truth，s是一个binary value，如果这个点正好是一个object的corner那他=1，否则=0，但是这里$y_{cij}$是0-1之间的数，是用基于ground truth角点的高斯分布计算得到，因此距离ground truth比较近的$(i^{'}, j^{'})$点的$y_{ci^{'}j^{'}}$值接近1。这样做是因为靠近ground truth的误检角点组成的预测框仍会和ground truth有较大的重叠面积，也能很好的框住原物体。

$$L_{det}=\frac{-1}{N}\sum_{c=1}^{C}\sum_{i=1}^{H}\sum_{j=1}^{W}\left\{\begin{matrix} (1-p_{cij})^{\alpha} log(p_{cij}) & if y_{cij}=1 \\ (1-y_{cij})^{\beta}(p_{cij})^{\alpha}log(1-p_{cij}) & Otherwise \end{matrix}\right.$$

计算对每个channel在heatmap上的每个位置的损失的和。如果这是一个正样本点（$y_{cij}=1$），那么使用focal loss计算损失，更多的关注难样本。如果不是，那么在focal loss的基础上加上$(1-y_{cij})^\beta$这一项，控制我们分配的标签对整体损失的影响。可以看到如果$y_{cij}$很接近1，这一项损失是接近0的，也就是说我们鼓励将这里的$p_{cij}$预测为1，这种soft的操作看起来就很舒服，注意使用object的数目$N$进行归一化。
#### Embedding Loss

论文中利用该损失函数来减小同一物体bounding box左上角和右下角embedding的距离，根据embedding的距离大小进行聚类，增大不同物体bounding box左上角和右下角embedding的距离，即$\tilde{E} \in R^{W/R \times H/R \times 1}$。

$$L_{pull}=\frac{1}{N}\sum_{k=1}^{N}[(e_{t_k}-e_k)^2+(e_{b_k}-e_k)^2]$$

$$L_{push}=\frac{1}{N(N-1)}\sum_{k=1}^{N}\sum_{i=1,j\neq 1}^{N}max(0, \triangle -|e_{b_k}-e_j|)$$

这两个损失函数得目的是pull损失使得类内距离尽可能小，push损失使得类间距离尽可能大。$e_{t_k}, e_{b_k, e_k}$分别是对象$k$预测的的右上角，左下角和平均的embedding，显然这里也只是存在object的地方才会计算embedding损失。
#### Offsets Loss

对于heatmap被downsample至原来的$\frac{1}{n}$后，还想继续upsample回去的话会造成精度的损失，这会严重影响到小物体框的位置精度，所以作者采用了offsets来缓解这种问题。对于输出值offset，$\tilde{O} \in R^{W/R \times H/R \times 2}$，我们使用卷积网络之后会将$(x, y)$位置的pixel映射到$([x/n], [y/n])$，而此时因为我们在特征图的每个pixel产生corner的预测，把他映射回去肯定会因为取整造成一定的误差（越小的box造成的误差越大），这个offset就是为了缓和这个取整误差：

$$o_k=(\frac{x_k}{n}-\left \lfloor \frac{x_k}{n} \right \rfloor, \frac{y_k}{n}-\left \lfloor \frac{y_k}{n} \right \rfloor)$$

每个pixel都会预测一对偏移值，但是只有存在object的pixel才会计算偏移误差。

$$L_{off}=\frac{1}{N}\sum_{k=1}^{N}{SmoothL1Loss}(o_k, \hat{o_k})$$

---

# CenterNet

但是CornerNet仍存在group环节带来了较大计算量，因此在此基础上出现了CenterNet。

CenterNet直接预测bbx的中心点，其他特征如大小、3D位置、方向，甚至姿态可以使用中心点位置的图像特征进行回归。将目标检测当成了关键点估计得任务来做，使用FCN将图像变成heatmap，峰值处就是我们想要的关键点。CenterNet的输出分辨率的下采样因子是4，比起其他的目标检测框架算是比较小的，因为centernet没有采用FPN结构，因此所有中心点要在一个Feature map上出，因此分辨率不能太低。

总之，CenterNet结构十分简单，直接检测目标的中心点和大小，是真正意义上的anchor-free。

## Network Architecture

论文中CenterNet提到了三种用于目标检测的网络，这三种网络都是编码解码(encoder-decoder)的结构：

1. Resnet-18 with up-convolutional layers : 28.1% coco and 142 FPS
2. DLA-34 : 37.4% COCOAP and 52 FPS
3. Hourglass-104 : 45.1% COCOAP and 1.4 FPS

每个网络内部的结构不同，但是在模型的最后输出部分都是加了三个网络构造来输出预测值，默认是80个类、2个预测的中心点坐标、2个中心点的偏置。


![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-4.png)

在整个训练的流程中，CenterNet学习了CornerNet的方法。对于每个标签图(ground truth)中的某一类，我们要将真实关键点(true keypoint) 计算出来用于训练，中心点的计算方式：$p=(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2})$，对于下采样后的坐标设为：$\tilde{p}=\left \lfloor \frac{p}{R} \right \rfloor$，，其中 R 是文中提到的下采样因子4。所以我们最终计算出来的中心点是对应低分辨率的中心点。

然后我们对图像进行标记，在下采样的[128,128]图像中将ground truth point以下采样的形式，用一个高斯滤波：

$$Y_{xyc}=exp(-\frac{(x-\tilde{p}_x)^2+(y-\tilde{p}_y)^2}{2\sigma ^2_p})$$

来将关键点分布到特征图上。

## Loss Function

### Center Points Loss Function

$$L_k=\frac{-1}{N}\sum_{xyc}\left\{\begin{matrix} (1-\hat{Y}_{xyc})^\alpha log(\hat{Y}_{xyc}) & if Y_{xyc}=1 \\ (1-\hat{Y}_{xyc})^{\beta}(\hat{Y}_{xyc})^{\alpha}log(1-\hat{Y}_{xyc}) & Otherwise \end{matrix}\right.$$

其中$\alpha$和$\beta$是Focal Loss的超参数，在这篇论文中分别是2和4，$N$是图像的关键点数量，用于将所有的positive focal loss标准化为1。这个损失函数是Focal Loss的修改版，适用于CenterNet。

而在CenterNet中，每个中心点对应一个目标的位置，不需要进行overlap的判断。那么怎么去减少negative center pointer的比例呢？CenterNet是采用Focal Loss的思想，在实际训练中，中心点的周围其他点(negative center pointer)的损失则是经过衰减后的损失(上文提到的)，而目标的长和宽是经过对应当前中心点的w和h回归得到的。

### Offset Loss

因为上文中对图像进行了$R=4$的下采样，这样的特征图重新映射到原始图像上的时候会带来精度误差，因此对于每一个中心点，额外采用了一个local offset 去补偿它。所有类c的中心点共享同一个offset prediction，这个偏置值(offset)用L1 loss来训练： 

$$L_{off}=\frac{1}{N}\sum_p |\hat{O}_{\tilde{p}}-(\frac{p}{R}-\tilde{p})|$$

这个偏置损失是可选的，我们不使用它也可以，只不过精度会下降一些。这部分跟CornerNet一致。

### Size Loss

假设目标$k$坐标为$(x_1^{(k)}, y_1^{(k)}, x_2^{(k)}, y_2^{(k)})$，所属类别为$c$，那它的中心点坐标为$p_k=(\frac{x_1^{(k)}+x_2^{(k)}}{2}, \frac{y_1^{(k)}+y_2^{(k)}}{2})$。我们使用关键点预测$\hat{Y}$去预测所有中心点，然后对每个目标$k$的size做回归，最终得到$s_k=(x_2^{(k)}-x_1^{(k)}, y_2^{(k)}-y_1^{(k)})$，这个值是在训练前提前计算出来的，是进行了下采样之后的长宽值。

为了减少回归的难度，这里使用$\hat{S}\in R^{\frac{W^{'}}{R}\times \frac{H}{R} \times 2}$作为预测值，使用L1损失函数，与之前的$L_{off}$损失一样：

$$L_{size}=\frac{1}{N}\sum_{k=1}^N |\hat{S}_{p_k}-s_k|$$

## Process 

在预测阶段，首先针对一张图像进行下采样，随后对下采样后的图像进行预测，对于每个类在下采样的特征图中预测中心点，然后将输出图中的每个类的热点单独地提取出来。具体怎么提取呢？就是检测当前热点的值是否比周围的八个近邻点(八方位)都大(或者等于)，然后取100个这样的点，采用的方式是一个3x3的MaxPool，类似于anchor-based检测中nms的效果。

这里假设$\hat{p}_c$为检测到的点，

$$\hat{p}=\{(\hat{x}_i, \hat{y}_i)\}_{i=1}^n$$

代表$c$类中检测到的一个点。每个关键点的位置用整型坐标表示$(x_i, y_i)$，然后使用$\hat{Y}_{x_i y_i c}$表示当前点的confidence，随后使用坐标来产生标定框：

$$(\hat{x}_i+\delta \hat{x}_i-\frac{\hat{w}_i}{2}, \hat{y}_i+\delta \hat{y}_i-\frac{\hat{w}_i}{2}, \hat{x}_i+\delta \hat{x}_i-\frac{\hat{w}_i}{2}, \hat{y}_i+\delta \hat{y}_i-\frac{\hat{w}_i}{2})$$

其中$(\delta \hat{x}_i, \delta \hat{y}_i)=\hat{O}\hat{x}_i, \hat{y}_i$，是当前点对应原始图像的偏置点，$(\hat{w}_i, \hat{h}_i)=\hat{S} \hat{x}_i, \hat{y}_i$代表预测出来当前点对应目标的长宽。

下图展示网络模型预测出来的中心点、中心点偏置以及该点对应目标的长宽：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-5.png)

那最终是怎么选择的，最终是根据模型预测出来的$\hat{Y}\in [0, 1]^{\frac{W}{R} \times \frac{H}{R} \times C}$值，也就是当前中心点存在物体的概率值，代码中设置的阈值为0.3，也就是从上面选出的100个结果中调出大于该阈值的中心点作为最终的结果。

## Conclusion

### Advantage

1. 设计模型的结构比较简单，不仅对于two-stage，对于one-stage的目标检测算法来说该网络的模型设计也是优雅简单的；
2. 该模型的思想不仅可以用于目标检测，还可以用于3D检测和人体姿态识别；
3. 虽然目前尚未尝试轻量级的模型，但是可以猜到这个模型对于嵌入式端这种算力比较小的平台还是很有优势的。
### Disadvantage

1. 在实际训练中，如果在图像中，同一个类别中的某些物体的GT中心点，在下采样时会挤到一块，也就是两个物体在GT中的中心点重叠了，CenterNet对于这种情况也是无能为力的，也就是将这两个物体的当成一个物体来训练(因为只有一个中心点)。同理，在预测过程中，如果两个同类的物体在下采样后的中心点也重叠了，那么CenterNet也是只能检测出一个中心点，不过CenterNet对于这种情况的处理要比faster-rcnn强一些的，具体指标可以查看论文相关部分。
2. 有一个需要注意的点，CenterNet在训练过程中，如果同一个类的不同物体的高斯分布点互相有重叠，那么则在重叠的范围内选取较大的高斯点。

---
# Reference

- [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/pdf/1808.01244)
- [CornerNet Code](https://github.com/princeton-vl/CornerNet)
- [CenterNet: Objects as Points](https://arxiv.org/pdf/1904.07850)
- [CenterNet Code](https://github.com/see--/keras-centernet)
