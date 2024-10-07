---

layout: post
tags: [Deep Learning, Batch Normalization]
title: Rethink BachNorm and GroupNorm
date: 2021-11-05
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

# Rethinking BatchNorm

在BatchNorm广泛应用之后，关于BN的一些思考也被提出，希望能从bacth本身的采样等方法里探讨，不同的batch会有什么样的不同效果。详见ref[1]。

本文简述其中涉及的四大实验，每个实验涉及一些子结论。

BatchNorm相对于其他算子来说，主要的不同在于BN是对batch数据进行操作的。BN在batch数据中进行统计量计算，而其他算子一般都是独立处理单个样本的。因此影响BN的输出不仅仅取决于单个样本的性质，还取决于batch的采样方式。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-1.jpeg)

如图所示，左右各举例了三种batch采样方式。

其中左图三种batch采样方式分别为：

    - entire dataset；
    - mini-batches；
    - subset of mini-batches

右图三种batch采样方式分别为：

    - entire domain；
    - each domain；
    - mixture of each domain

## PreciseBN

BN中统计量的计算默认使用EMA方法，但是作者实验发现EMA会导致模型性能次优，然后提出了PreciseBN方法，近似将整个训练集统计量作为一个batch。

EMA，exponential moving average，指数移动平均，可以用来有效地计算总体统计，但是可能会存在一些不良情况。

$$\mu_{EMA}\leftarrow\lambda\mu_{EMA}+(1-\lambda)\mu_\beta$$

$$\sigma^2_{EMA}\leftarrow\lambda\sigma^2_{EMA}+(1-\lambda)\sigma^2_{\beta}$$

不难看出$\mu_{EMA}$和$\sigma^2_{EMA}$根据权重值$\lambda$不断更新，但是导致次优解的原因也是在于：

1. 当$\lambda$太大时，统计量收敛速度变慢；
2. 当$\lambda$太小时，会受到newest的几个mini-batch影响更大，统计量无法表示整个training set samples的统计量。

为了得到整个训练集更加精确的统计量，PreciseBN采用了两点小技巧：

1. 将相同模型用于多个mini-batches来收集batch统计量
2. 将多个batch收集的统计量聚合成一个population统计量

比如有N个样本需要通过数量为的Bmini-batch进行PreciseBN统计量计算，那么需要计算$k=\frac{N}{B}$次，统计量聚合公式为：

$$\mu_{pop}=E[\mu_\beta]$$

$$\sigma^2_{pop}=E[\mu^2_\beta+\sigma^2_\beta]-E[\mu_\beta]^2$$


相比于EMA，PreciseBN有两点重要的属性：

1. PreciseBN的统计量是通过相同模型计算得到的，而EMA是通过多个历史模型计算得到的。
2. PreciseBN的所有样本的权重是相同的，而EMA不同样本的权重是不同的。


![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-2.jpeg)

最后实验得出结论：

1. 推理时使用PreciseBN会更加稳定。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-3.jpeg)
2. 大batch训练对EMA影响更大。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-4.jpeg)
3. PreciseBN只需要$10^3$~$10^4$个样本可以得到近似最优。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-5.jpeg)
4. 小batch会产生统计量积累错误。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-6.jpeg)
## Batch in Training & Testing

BN在训练和测试中行为不一致：训练时，BN的统计量来自mini-batch；测试时，BN的统计量来自population。这部分主要探讨了BN行为不一致对模型性能的影响，并且提出消除不一致的方法提升模型性能。

为了避免混淆，将SGD batch size或者total batch size定义为所有GPU上总的batch size大小，将normalization batch size定义为单个GPU上的batch size大小。

normalization batch size对training noise和train-test inconsistency有着直接影响：使用更大的batch，mini-batch统计量越接近population统计量，从而降低training noise和train-test inconsistency。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-7.jpeg)

- **Training noise**：当normalization batch size非常小时，单个样本会受到同一个min-batch样本的严重影响，导致训练精度较差，优化困难。
- **Generalization gap**：随着normalization batch size的增加，mini-batch的验证集和训练集的之间的泛化误差会增大，这可能是由于training noise和train-test inconsistency没有正则化。
- **Train-test inconsistency**：在小batch下，mini-batch统计量和population统计量的不一致是影响性能的主要因素。当normalization batch size增大时，细微的不一致可以提供正则化效果减少验证误差。在mini-batch为32~128之间时，正则化达到平衡，模型性能最优。

为了保持train和test的BN统计量一致，作者提出了两种方法来解决不一致问题:

1. 一种是推理的时候使用mini-batch统计量，
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-8.jpeg)
2. 另一种是训练的时候使用population batch统计量。这里作者采用FrozenBN的方法，先选择第80个epoch模型，然后将所有BN替换成FrozenBN，然后训练20个epoch。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-9.jpeg)

## Batch from different Domain

BN的训练过程分为：
1. 通过SGD学习features，
2. 由这些features得到population统计量。

由于BN多了一个population统计阶段，导致训练和测试之间的domain shift。当数据来自多个doman时，SGD training、population statistics training和testing三个步骤的domain gap都会对泛化性造成影响。

实验主要探究了两种使用场景：
1. 模型在一个domain上进行训练，然后在其他domain上进行测试；
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-10.jpeg)
2. 模型在多个domain上进行训练。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-11.jpeg)

结论：
1. 当存在显著的domain shift时，模型使用评估domain的population统计量会得到更好的结果，可以缓解训练测试的不一致。
2. SGD training、population statistics training和testing保持一致是非常重要的，并且全部使用domain-specific能取得最好的效果。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-12.jpeg)
## Information Leaking within a Batch

BN在使用中还存在一种information leakage现象，因为BN是对mini-batch的样本计算统计量的，导致在样本进行独立预测时，会利用mini-batch内其他样本的统计信息。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-13.jpeg)

作者实验发现，当使用random采样的mini-batch统计量时，验证误差会增加，当使用population统计量时，验证误差会随着epoch的增加逐渐增大，验证了BN信息泄露问题的存在。
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-14.jpeg)

解决方法：
1. 使用SyncBN，来弱化mini-batch内样本之间的相关性。
2. 在进入head之前在GPU之间随机打乱RoI features，这给每个GPU分配了一个随机的样本子集来进行归一化，同时也削弱了min-batch样本之间的相关性。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-15.jpeg)

实验结果表明，shuffling和SyncBN都能有效地处理信息泄漏，使得head在测试时能够很好地泛化。在速度方面，我们注意到shuffling需要更少的跨gpu同步，但是shuffling每次传输的数据比SyncBN多。因此，shuffling和SyncBN的相对效率跟具体模型架构相关。

---

# Group Normalization

由于BN存在于上文所提到的一些基于batch的问题，所以Group Normalization是Face book AI research（FAIR）吴育昕-何恺明联合推出用于改进和替代BN的方法。

## What is GN

GN本质上仍是归一化，但是它灵活的避开了BN的问题，同时又不同于Layer Norm，Instance Norm，四者的关系如下：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-16.jpeg)

从左到右以此是BN，LN，IN，GN。
深度网络中的数据维度一般是[N, C, H, W]或者[N, H, W，C]格式，N是batch size，H/W是feature的高/宽，C是feature的channel，压缩H/W至一个维度，其三维的表示如上图，假设单个方格的长度是1，那么其表示的是[6, 6，*, * ]。

- **BN**在batch的维度上norm，归一化维度为[N，H，W]，对batch中对应的channel归一化；
- **LN**避开了batch维度，归一化的维度为[C，H，W]；
- **IN**归一化的维度为[H，W]；
- 而**GN**介于LN和IN之间，其首先将channel分为许多组（group），对每一组做归一化，及先将feature的维度由[N, C, H, W]reshape为[N, G，C//G , H, W]，归一化的维度为[C//G , H, W]

事实上，GN的极端情况就是LN和I N，分别对应G等于C和G等于1，作者在论文中给出G设为32较好。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-17.jpeg)

给出的实践代码：

    def GroupNorm(x, gamma, beta, G, eps=1e-5):
        # x: input features with shape [N,C,H,W]
        # gamma, beta: scale and offset, with shape [1,C,1,1]
        # G: number of groups for GN
        N, C, H, W = x.shape
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, [N, C, H, W])
        return x * gamma + beta

其中$\beta$和$\gamma$参数是norm中可训练参数，表示平移和缩放因子。

## Why GN works？

传统角度来讲，在深度学习没有火起来之前，提取特征通常是使用SIFT，HOG和GIST特征，这些特征有一个共性，都具有按group表示的特性，每一个group由相同种类直方图的构建而成，这些特征通常是对在每个直方图（histogram）或每个方向（orientation）上进行组归一化（group-wise norm）而得到。而更高维的特征比如VLAD和Fisher Vectors(FV)也可以看作是group-wise feature，此处的group可以被认为是每个聚类（cluster）下的子向量sub-vector。

从深度学习上来讲，完全可以认为卷积提取的特征是一种非结构化的特征或者向量，拿网络的第一层卷积为例，卷积层中的的卷积核filter1和此卷积核的其他经过transform过的版本filter2（transform可以是horizontal flipping等），在同一张图像上学习到的特征应该是具有相同的分布，那么，具有相同的特征可以被分到同一个group中，按照个人理解，每一层有很多的卷积核，这些核学习到的特征并不完全是独立的，某些特征具有相同的分布，因此可以被group。

导致分组（group）的因素有很多，比如频率、形状、亮度和纹理等，HOG特征根据orientation分组，而对神经网络来讲，其提取特征的机制更加复杂，也更加难以描述，变得不那么直观。另在神经科学领域，一种被广泛接受的计算模型是对cell的响应做归一化，此现象存在于浅层视觉皮层和整个视觉系统。

作者基于此，提出了组归一化（Group Normalization）的方式，且效果表明，显著优于BN、LN、IN等。

GN的归一化方式避开了batch size对模型的影响，特征的group归一化同样可以解决Internal Covariate Shift的问题，并取得较好的效果。

## How it works？

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-18.jpeg)

以resnet50为base model，batchsize设置为32在imagenet数据集上的训练误差（左）和测试误差（右）。GN没有表现出很大的优势，在测试误差上稍大于使用BN的结果。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-19.jpeg)

可以很容易的看出，GN对batch size的鲁棒性更强。

同时，作者以VGG16为例，分析了某一层卷积后的特征分布学习情况，分别根据不使用Norm 和使用BN，GN做了实验，实验结果如下：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211106-20.jpeg)

统一batch size设置的是32，最左图是不使用norm的conv5的特征学习情况，中间是使用了BN结果，最右是使用了GN的学习情况，相比较不使用norm，使用norm的学习效果显著，而后两者学习情况相似，不过更改小的batch size后，BN是比不上GN的。

作者同时做了实验展示了GN在object detector/segmentation 和video classification上的效果，详情可见原文，此外，作者在paper最后一节中大致探讨了discussion and future work , 实乃业界良心。

---

# Reference

- [Rethinking "Batch" in BatchNorm - arXiv](https://arxiv.org/abs/2105.07576)
- [Group Normalization - arXiv](https://arxiv.org/abs/1803.08494)