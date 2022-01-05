---

layout: post
categories: [Computer Vision]
tags: [Detection, Deep Learning]

---

# Why Anchor Free?

随着anchor在检测算法中的应用，不管是one or two stage的检测模型，都会在图片上放置密密麻麻尺寸不一的anchors，用来检测全图各个角落大小不一的目标物体。但是anchor based model有两个不足之处：

- Anchor amount过大，导致计算复杂度过高，对于绝大部分情况，只有其中很小一部分anchor能成功匹配到ground truth，而大量anchor作为负样本被丢弃。

- 整体引入大量hyper-parameters，影响模型性能。

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

cornerNet的网络结构主要分为以下几个部分

1. Backbone: Hourglass Network；
2. Head: 二分支输出 Top-left corners 和 Bottom-right corners，每个分支包含了各自的corner pooling以及三分支输出。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-3.png)

cornerNet借用了hourglass network作为他的backbone特征提取网络，这个hourglass network通常被用在姿态估计任务中，是一种呈沙漏状的downsampling 和 upsampling组合，为两个沙漏模块（hourglass module）头尾相连的效果。

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


---
# Reference

- [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/pdf/1808.01244)
- [CornerNet Code](https://github.com/princeton-vl/CornerNet)
- [CenterNet: Objects as Points](https://arxiv.org/pdf/1904.07850)
- [CenterNet Code](https://github.com/see--/keras-centernet)
