---

layout: post
categories: [Algorithm]
tags: [Deep Learning, Batch Normalization]

---

# What is Batch Normalization？

**Batch Normalization**（以下用BN简称代替）是为了解决Deep Learning中*Internal Covariate Shift（内部协变量移位）*问题，而针对一个batch中的数据进行归一标准化的方法。作用是可以使用更加flexible的学习率learning rate，得到更快的学习速率，同时不会过于依赖模型的初始化initialization。其中所谓的Internal Covariate Shift是指在deep neural network当中，随着层数的加深，每一层layer都关联着其上下层，其input实上一层layer的output，而其output又是下一层layer的input，而在随机梯度下降学习过程中，随着每层layer参数的更新，其对应的input/output分布也在随时起着微小的变化，这种变化会随着层数深度的加深而逐步积累，从而使得整个学习过程难度加大。

# Introduce BN

BN的整体思路其实是一种*whiten白化*操作，即将输入的数据分布转换到均值$\mu=0$，方差$\sigma=1$的正态分布，能有效加快整个network的收敛，即用较少的学习步骤获得相同的推理精度。直观上的现象就是缓解梯度消失，支持更大的学习率。因为大学习率往往导致反向时梯度困在局部最小值处，BN可以避免神经网络层中很小的参数变动在层数加深的过程中会积聚造成很大的影响。

一般会有以下两种对不同阶段BN的操作方法：

1. 在forward process前向传递过程中进行BN，而在backward process后向传递的过程中无视其他样本对当前样本的影响。举个例子，对于某一层的输入$\mu$和学习到的偏移量$b$，进行白化$\hat{x}=x-E[x]$，其中$x=\mu+b$，进行梯度更新时$b=b+\Delta b$，$\Delta b=-\frac{\partial l}{\partial b}$，而输出$\hat{x}'=\mu+(b+\Delta b)-E[\mu+(b+\Delta b)]=\mu+b-E[\mu+b]$，可以看出该层的输出和损失没有任何变化，但随着更新$b$会趋近于无穷。所以在normalization时要考虑优化的过程。
2. 在前向传播进行normalization，反向传播考虑其他样本数据对于当前样本的影响。如果将normalization过程看做一次变化，$\hat{x}=Norm(x, \chi)$。反向传播时考虑其他样本数据对当前样本的影响，需要计算两个导数：$\frac{\partial Norm(x, \chi)}{\partial x}$和$\frac{\partial Norm(x, \chi)}{\partial \chi}$，其中第二个计算的复杂度较大。

为此有两处做了简化：

1. 对每个特征独立的进行normalization。

2. 简单的normalization每一层的输入降低了模型的表现能力，比如norm后通过sigmoid激活函数数据分布在中间线性的区域，所以加入可学习的参数进行scale and shift。

# Method

如果batch size为m，则在前向传播时每个节点都有m个输出，对每个节点的m个输出进行归一化。实现如下图所示，可以分为两步：

 - Standardization：对m个$x$进行标准化得到zero mean unit variance的$\hat{x}$
 - Scale and shift：对$\hat{x}$进行缩放和平移，得到最终的分布$y$，具有新的均值$\beta$和方差$\gamma$，其中$\beta$和$\gamma$是作为网络训练过程中需要被训练的参数形式出现，其最终取值会在训练过程中不断变化得到。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211006-1.jpg)

针对Batch Normalization Transform，总体可以写为如下公式，所以，无论$x_i$原本的均值和方差是多少，通过Batch Norm后其均值和方差分别变为待学习的$\beta$和$\gamma$。

$y^{(b)}_i=BN(x_i)^{b}=\gamma\cdot(\frac{x_i^{b}-\mu(x_i)}{\sqrt{}\sigma(x_i)^2})+\beta$

整体变化流程如下图所示：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211006-3.jpg)

---

# Training & Testing

在训练时，$\mu$，$\sigma$是当前mini batch的统计量，随着batch的不同一直在变化。在预测阶段，我们希望模型的输出只和输入相关，所以$\mu$，$\sigma$应该固定，可以采用移动平均来计算$\mu$，$\sigma$。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211006-2.jpg)

---

# Effect

1. **可以使用更大的学习率**，训练过程更加稳定，极大提高了训练速度。
2. **可以将bias置为0**，因为Batch Normalization的Standardization过程会移除直流分量，所以不再需要bias。
3. **对权重初始化不再敏感**，通常权重采样自0均值某方差的高斯分布，以往对高斯分布的方差设置十分重要，有了Batch Normalization后，对与同一个输出节点相连的权重进行放缩，其标准差σσ也会放缩同样的倍数，相除抵消。
4. **对权重的尺度不再敏感**，理由同上，尺度统一由$\gamma$参数控制，在训练中决定。
5. **深层网络可以使用sigmoid和tanh了**，理由同上，BN抑制了梯度消失。
6. **Batch Normalization具有某种正则作用**，不需要太依赖dropout，减少过拟合。

---

# Summary

- **卷积层如何使用Batch Norm？**

1个卷积核产生1个feature map，1个feature map有1对$\gamma$和$\beta$参数，同一batch同channel的feature map共享同一对$\gamma$和$\beta$参数，若卷积层有n个卷积核，则有n对$\gamma$和$\beta$参数。

- **没有scale and shift过程可不可以?**

可以，但网络的表达能力会下降。对输入进行scale and shift，有利于分布与权重的相互协调。

- **BN层放在ReLU前面还是后面？**

原paper建议将BN层放置在ReLU前，因为ReLU激活函数的输出非负，不能近似为高斯分布。实验表明，放在前后的差异似乎不大，甚至放在ReLU后还好一些。

- **BN层为什么有效？**

    - BN层让损失函数更平滑：加BN层后，损失函数的landscape(loss surface)变得更平滑，相比高低不平上下起伏的loss surface，平滑loss surface的梯度预测性更好，可以选取较大的步长。
    - BN更有利于梯度下降：没有BN层的，其loss surface存在较大的高原，有BN层的则没有高原，而是山峰，因此更容易下降。
    - 没有BN层的情况下，网络没办法直接控制每层输入的分布，其分布前面层的权重共同决定，或者说分布的均值和方差“隐藏”在前面层的每个权重中，网络若想调整其分布，需要通过复杂的反向传播过程调整前面的每个权重实现，BN层的存在相当于将分布的均值和方差从权重中剥离了出来，只需调整$\gamma$和$\beta$两个参数就可以直接调整分布，让分布和权重的配合变得更加容易。




# Reference

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [Batch Normalization详解 - shine-lee - 博客园 (cnblogs.com)](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/shine-lee/p/11989612.html)

- [How Does Batch Normalization Help Optimization?](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1805.11604)

- [An empirical analysis of the optimization of deep network loss surfaces (arxiv.org)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1612.04010)

- [Batch Normalization论文解读+详细面经_joyce_peng的博客-CSDN博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/joyce_peng/article/details/103163048)