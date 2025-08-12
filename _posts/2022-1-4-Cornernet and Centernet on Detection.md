---

layout: post
tags: [Detection, Deep Learning, Computer Vision]
title: CornerNet and CenterNet on Detection
date: 2022-01-04
author: Kyrie Chen
comments: true
toc: true
pinned: false
---

随着 anchor 在检测算法中的应用，不管是 one- 或 two-stage 的检测模型，都会在图片上放置密密麻麻、尺寸不一的 anchors，用来覆盖全图各个角落的目标。但 anchor-based 方法存在两点不足：

- Anchor 数量过大，计算与显存开销高。大多数 anchors 与 GT 不匹配，成为被丢弃的负样本；
- 大量超参数（尺度、长宽比、阈值等）与后处理 NMS，使训练与部署复杂且影响性能。
# CornerNet

---

# CornerNet

CornerNet 的论文题为 `Detecting Objects as Paired Keypoints`，核心思想是“以配对关键点表示目标框”。本文从以下几点展开：

1. 如何用 anchor-free 的方式表示目标；
2. 什么是 corner pooling；
3. 网络结构与损失函数；
4. 配对与后处理。

## Locate the Object

在 CornerNet 中，我们预测目标 bbox 的左上（top-left）与右下（bottom-right）两个角点作为关键点。为了解决“跨目标配对”问题，网络为每个角点学习一个 embedding（又称 tag），同一目标的两角 embedding 距离更近、不同目标更远。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-1.png)

作者选择 corner 而非 center 的原因：中心点需回归宽高（四边）才能确定框，而角点只需两条边；以 corner 表示 bbox 也更符合“关键点检测”的范式。

## Corner Pooling

Corner pooling 用于在边界处强化角点响应：以 top-left 为例，在每个像素上沿“向右”的水平方向与“向下”的垂直方向分别做 max pooling，再将二者相加，从而聚合“到边界为止”的极值特征；bottom-right 则相反方向。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-2.png)
## CornerNet Detail

### Network Architecture

CornerNet 的网络结构主要分为以下几部分：

1. Backbone: Hourglass Network；
2. Head: 二分支输出 Top-left corners 和 Bottom-right corners，每个分支包含了各自的corner pooling以及三分支输出。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-3.png)

CornerNet 使用 Hourglass 作为 backbone。Hourglass 常见于姿态估计，由对称的下采样与上采样模块组成，两端堆叠以增强表达能力。

在原有hourglass的基础上，作者做了以下的改进：

1. 在输入hourglass module之前，需要将图片分辨率降低为原来的1/4倍。本文采用了一个stride=2的7x7卷积和一个stride=2的残差单元进行图片分辨率降低。
2. 使用stride=2的卷积层代替max pooling进行downsample
3. 共进行5次downsample ,这5次downsample后的特征图通道为[256,384,384,384,512]
4. 采用最近邻插值的上采样（upsample),后面接两个残差单元

### Loss Function

#### Focal Loss（改造版）

首先是角点热图的分类损失。网络输出角点 heatmap，维度为 \(\tilde{M}\in\mathbb{R}^{\tfrac{W}{R}\times\tfrac{H}{R}\times C}\)。GT 不是二值而是基于角点位置绘制的高斯分布（软标签），以鼓励靠近 GT 的预测点。损失采用改造版 Focal Loss，对正样本与负样本进行不同衰减。

$$L_{det}=\frac{-1}{N}\sum_{c=1}^{C}\sum_{i=1}^{H}\sum_{j=1}^{W}\left\{\begin{matrix} (1-p_{cij})^{\alpha} log(p_{cij}) & if y_{cij}=1 \\ (1-y_{cij})^{\beta}(p_{cij})^{\alpha}log(1-p_{cij}) & Otherwise \end{matrix}\right.$$

计算对每个channel在heatmap上的每个位置的损失的和。如果这是一个正样本点（$y_{cij}=1$），那么使用focal loss计算损失，更多的关注难样本。如果不是，那么在focal loss的基础上加上$(1-y_{cij})^\beta$这一项，控制我们分配的标签对整体损失的影响。可以看到如果$y_{cij}$很接近1，这一项损失是接近0的，也就是说我们鼓励将这里的$p_{cij}$预测为1，这种soft的操作看起来就很舒服，注意使用object的数目$N$进行归一化。
#### Embedding Loss

论文同时引入 embedding 的 pull/push 损失：同一目标的两角向“中心 embedding”拉近，不同目标之间相互推远，\(\tilde{E}\in\mathbb{R}^{\tfrac{W}{R}\times\tfrac{H}{R}\times1}\)。

$$L_{pull}=\frac{1}{N}\sum_{k=1}^{N}[(e_{t_k}-e_k)^2+(e_{b_k}-e_k)^2]$$

$$L_{push}=\frac{1}{N(N-1)}\sum_{k=1}^{N}\sum_{i=1,j\neq 1}^{N}max(0, \triangle -|e_{b_k}-e_j|)$$

pull 使类内更紧，push 使类间更分离。\(e_{t_k}, e_{b_k}, e_k\) 分别为目标 \(k\) 的两角与均值 embedding，仅在目标处计算该项损失。
#### Offsets Loss

对于下采样后的 heatmap 回映射到原图会有取整误差（对小目标尤甚），因此引入 offset 分支 \(\tilde{O}\in\mathbb{R}^{\tfrac{W}{R}\times\tfrac{H}{R}\times2}\) 进行补偿：

$$o_k=(\frac{x_k}{n}-\left \lfloor \frac{x_k}{n} \right \rfloor, \frac{y_k}{n}-\left \lfloor \frac{y_k}{n} \right \rfloor)$$

每个pixel都会预测一对偏移值，但是只有存在object的pixel才会计算偏移误差。

$$L_{off}=\frac{1}{N}\sum_{k=1}^{N}{SmoothL1Loss}(o_k, \hat{o_k})$$

---

# CenterNet

CornerNet 在推理时需要对角点进行 grouping 与配对，带来一定计算与失败风险。CenterNet 在此基础上提出“以中心点表示目标”，显著简化流程。

CenterNet 直接预测 bbox 中心点，并回归宽高、offset 等属性；也可扩展到 3D、姿态等。其本质是“关键点检测”：用 FCN 产生 heatmap，局部极值即中心点。默认下采样因子 R=4；由于不使用 FPN，所有中心在同一特征图上预测，因而分辨率不能太低。

总之，CenterNet结构十分简单，直接检测目标的中心点和大小，是真正意义上的anchor-free。

## Network Architecture

论文中CenterNet提到了三种用于目标检测的网络，这三种网络都是编码解码(encoder-decoder)的结构：

1. ResNet‑18 with up-conv：28.1% COCO AP，142 FPS
2. DLA‑34：37.4% COCO AP，52 FPS
3. Hourglass‑104：45.1% COCO AP，1.4 FPS

各骨干不同，但头部统一输出三个分支：类别 heatmap、中心点偏置（offset）与宽高（wh）。默认 80 类、2 个坐标 offset、2 个尺寸值。


![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-4.png)

训练时将每个 GT 框的中心点投影到下采样特征图：\(p=(\tfrac{x_1+x_2}{2},\tfrac{y_1+y_2}{2})\), \(\tilde{p}=\left\lfloor p/R\right\rfloor\)。围绕 \(\tilde{p}\) 绘制高斯核，半径由目标尺寸与 IoU 阈值联合确定，以减少中心点冲突。

然后我们对图像进行标记，在下采样的[128,128]图像中将ground truth point以下采样的形式，用一个高斯滤波：

$$Y_{xyc}=exp(-\frac{(x-\tilde{p}_x)^2+(y-\tilde{p}_y)^2}{2\sigma ^2_p})$$

来将关键点分布到特征图上。

## Loss Function

### Center Points Loss Function（改造版 Focal）

$$L_k=\frac{-1}{N}\sum_{xyc}\left\{\begin{matrix} (1-\hat{Y}_{xyc})^\alpha log(\hat{Y}_{xyc}) & if Y_{xyc}=1 \\ (1-\hat{Y}_{xyc})^{\beta}(\hat{Y}_{xyc})^{\alpha}log(1-\hat{Y}_{xyc}) & Otherwise \end{matrix}\right.$$

其中$\alpha$和$\beta$是Focal Loss的超参数，在这篇论文中分别是2和4，$N$是图像的关键点数量，用于将所有的positive focal loss标准化为1。这个损失函数是Focal Loss的修改版，适用于CenterNet。

每个中心点唯一对应一个目标，不再需要 IoU 匹配。为缓解负样本数量过大，采用改造版 Focal Loss 对非中心区域进行衰减。宽高由对应位置的 \(w,h\) 分支回归得到。

### Offset Loss

由于下采样带来量化误差，对每个中心点回归局部 offset（所有类别共享），使用 L1 损失： 

$$L_{off}=\frac{1}{N}\sum_p |\hat{O}_{\tilde{p}}-(\frac{p}{R}-\tilde{p})|$$

这个偏置损失是可选的，我们不使用它也可以，只不过精度会下降一些。这部分跟CornerNet一致。

### Size Loss

假设目标$k$坐标为$(x_1^{(k)}, y_1^{(k)}, x_2^{(k)}, y_2^{(k)})$，所属类别为$c$，那它的中心点坐标为$p_k=(\frac{x_1^{(k)}+x_2^{(k)}}{2}, \frac{y_1^{(k)}+y_2^{(k)}}{2})$。我们使用关键点预测$\hat{Y}$去预测所有中心点，然后对每个目标$k$的size做回归，最终得到$s_k=(x_2^{(k)}-x_1^{(k)}, y_2^{(k)}-y_1^{(k)})$，这个值是在训练前提前计算出来的，是进行了下采样之后的长宽值。

为了减少回归的难度，这里使用$\hat{S}\in R^{\frac{W^{'}}{R}\times \frac{H}{R} \times 2}$作为预测值，使用L1损失函数，与之前的$L_{off}$损失一样：

$$L_{size}=\frac{1}{N}\sum_{k=1}^N |\hat{S}_{p_k}-s_k|$$

## Process 

推理阶段，先对热图做 3×3 max-pooling 获得局部极值（近似 NMS），每类取 top‑K（如 100）中心点作为候选。

这里假设$\hat{p}_c$为检测到的点，

$$\hat{p}=\{(\hat{x}_i, \hat{y}_i)\}_{i=1}^n$$

代表 $c$ 类中检测到的一个点。每个关键点的位置用整型坐标表示 $(x_i, y_i)$，其置信度为 $\hat{Y}_{x_i y_i c}$。bbox 由坐标、offset 与宽高回构：

$$(\hat{x}_i+\delta \hat{x}_i-\tfrac{\hat{w}_i}{2},\ \hat{y}_i+\delta \hat{y}_i-\tfrac{\hat{h}_i}{2},\ \hat{x}_i+\delta \hat{x}_i+\tfrac{\hat{w}_i}{2},\ \hat{y}_i+\delta \hat{y}_i+\tfrac{\hat{h}_i}{2})$$

其中 $(\delta \hat{x}_i, \delta \hat{y}_i)=\hat{O}_{x_i y_i}$ 为 offset，$(\hat{w}_i, \hat{h}_i)=\hat{S}_{x_i y_i}$ 为回归的宽高。

下图展示网络模型预测出来的中心点、中心点偏置以及该点对应目标的长宽：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220104-5.png)

最终根据 $\hat{Y}\in[0,1]^{\tfrac{W}{R}\times\tfrac{H}{R}\times C}$ 的置信度与阈值（如 0.3）筛选中心点，回构对应 bbox 后输出结果。

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
