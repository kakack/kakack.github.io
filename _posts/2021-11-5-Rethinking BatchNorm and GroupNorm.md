---

layout: post
categories: [Algorithm]
tags: [Deep Learning, Batch Normalization]

---

# Rethinking BatchNorm

在BatchNorm广泛应用之后，关于BN的一些思考也被提出，希望能从bacth本身的采样等方法里探讨，不同的batch会有什么样的不同效果。详见[ref[1]](#1)。

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

$\mu_{EMA}\leftarrow\lambda\mu_{EMA}+(1-\lambda)\mu_\beta$

$\sigma^2_{EMA}\leftarrow\lambda\sigma^2_{EMA}+(1-\lambda)\sigma^2_{\beta}$

不难看出$\mu_{EMA}$和$\sigma^2_{EMA}$根据权重值$\lambda$不断更新，但是导致次优解的原因也是在于：

1. 当$\lambda$太大时，统计量收敛速度变慢；
2. 当$\lambda$太小时，会受到newest的几个mini-batch影响更大，统计量无法表示整个training set samples的统计量。

为了得到整个训练集更加精确的统计量，PreciseBN采用了两点小技巧：

1. 将相同模型用于多个mini-batches来收集batch统计量
2. 将多个batch收集的统计量聚合成一个population统计量

比如有N个样本需要通过数量为的Bmini-batch进行PreciseBN统计量计算，那么需要计算$k=\frac{N}{B}$次，统计量聚合公式为：

$\mu_{pop}=E[\mu_\beta]$

$\sigma^2_{pop}=E[\mu^2_\beta+\sigma^2_\beta]-E[\mu_\beta]^2$


相比于EMA，PreciseBN有两点重要的属性：

1. PreciseBN的统计量是通过相同模型计算得到的，而EMA是通过多个历史模型计算得到的。
2. PreciseBN的所有样本的权重是相同的，而EMA不同样本的权重是不同的。

最后实验得出结论：

1. 推理时使用PreciseBN会更加稳定。
2. 大batch训练对EMA影响更大。
3. PreciseBN只需要$10^3$~$10^4$个样本可以得到近似最优。
4. 小batch会产生统计量积累错误。

## Batch in Training & Testing

BN在训练和测试中行为不一致：训练时，BN的统计量来自mini-batch；测试时，BN的统计量来自population。这部分主要探讨了BN行为不一致对模型性能的影响，并且提出消除不一致的方法提升模型性能。

为了避免混淆，将SGD batch size或者total batch size定义为所有GPU上总的batch size大小，将normalization batch size定义为单个GPU上的batch size大小。

normalization batch size对training noise和train-test inconsistency有着直接影响：使用更大的batch，mini-batch统计量越接近population统计量，从而降低training noise和train-test inconsistency。

- **Training noise**：当normalization batch size非常小时，单个样本会受到同一个min-batch样本的严重影响，导致训练精度较差，优化困难。
- **Generalization gap**：随着normalization batch size的增加，mini-batch的验证集和训练集的之间的泛化误差会增大，这可能是由于training noise和train-test inconsistency没有正则化。
- **Train-test inconsistency**：在小batch下，mini-batch统计量和population统计量的不一致是影响性能的主要因素。当normalization batch size增大时，细微的不一致可以提供正则化效果减少验证误差。在mini-batch为32~128之间时，正则化达到平衡，模型性能最优。

为了保持train和test的BN统计量一致，作者提出了两种方法来解决不一致问题:

1. 一种是推理的时候使用mini-batch统计量，
2. 另一种是训练的时候使用population batch统计量。这里作者采用FrozenBN的方法，先选择第80个epoch模型，然后将所有BN替换成FrozenBN，然后训练20个epoch。

## Batch from different Domain

BN的训练过程分为：
1. 通过SGD学习features，
2. 由这些features得到population统计量。

由于BN多了一个population统计阶段，导致训练和测试之间的domain shift。当数据来自多个doman时，SGD training、population statistics training和testing三个步骤的domain gap都会对泛化性造成影响。

实验主要探究了两种使用场景：
1. 模型在一个domain上进行训练，然后在其他domain上进行测试；
2. 模型在多个domain上进行训练。

结论：
1. 当存在显著的domain shift时，模型使用评估domain的population统计量会得到更好的结果，可以缓解训练测试的不一致。
2. SGD training、population statistics training和testing保持一致是非常重要的，并且全部使用domain-specific能取得最好的效果。
## Information Leaking within a Batch

BN在使用中还存在一种information leakage现象，因为BN是对mini-batch的样本计算统计量的，导致在样本进行独立预测时，会利用mini-batch内其他样本的统计信息。

作者实验发现，当使用random采样的mini-batch统计量时，验证误差会增加，当使用population统计量时，验证误差会随着epoch的增加逐渐增大，验证了BN信息泄露问题的存在。

解决方法：
1. 使用SyncBN，来弱化mini-batch内样本之间的相关性。
2. 在进入head之前在GPU之间随机打乱RoI features，这给每个GPU分配了一个随机的样本子集来进行归一化，同时也削弱了min-batch样本之间的相关性。

实验结果表明，shuffling和SyncBN都能有效地处理信息泄漏，使得head在测试时能够很好地泛化。在速度方面，我们注意到shuffling需要更少的跨gpu同步，但是shuffling每次传输的数据比SyncBN多。因此，shuffling和SyncBN的相对效率跟具体模型架构相关。



---

# Group Normalization

---

# <p id="1">Reference </p>

- [Rethinking "Batch" in BatchNorm - arXiv](https://arxiv.org/abs/2105.07576)
- [Group Normalization - arXiv](https://arxiv.org/abs/1803.08494)