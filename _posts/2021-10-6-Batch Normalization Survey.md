---

layout: post
categories: [Algorithm]
tags: [Deep Learning, Batch Normalization]

---

# What is Batch Normalization？

**Batch Normalization**（以下用BN简称代替）是为了解决Deep Learning中*Internal Covariate Shift（内部协变量移位）*问题，而针对一个batch中的数据进行归一标准化的方法。作用是可以使用更加flexible的学习率learning rate，得到更快的学习速率，同时不会过于依赖模型的初始化initialization。其中所谓的Internal Covariate Shift是指在deep neural network当中，随着层数的加深，每一层layer都关联着其上下层，其input实上一层layer的output，而其output又是下一层layer的input，而在随机梯度下降学习过程中，随着每层layer参数的更新，其对应的input/output分布也在随时起着微小的变化，这种变化会随着层数深度的加深而逐步积累，从而使得整个学习过程难度加大。

# Introduce BN

BN的整体思路其实是一种*whiten（白化）*操作，即将输入的数据分布转换到均值$\mu=0$，方差$\sigma=1$的正态分布，能有效加快整个network的收敛，即用较少的学习步骤获得相同的推理精度。直观上的现象就是缓解梯度消失，支持更大的学习率。因为大学习率往往导致反向时梯度困在局部最小值处，BN可以避免神经网络层中很小的参数变动在层数加深的过程中会积聚造成很大的影响。

一般会有以下两种对不同阶段BN的操作方法：

1. 在forward process前向传递过程中进行BN，而在backward process后向传递的过程中无视其他样本对当前样本的影响。举个例子，对于某一层的输入$\mu$和学习到的偏移量$b$，进行白化$\hat{x}=x-E[x]$，其中$x=\mu+b$，进行梯度更新时$b=b+\Delta b$，$\Delta b=-\frac{\partial l}{\partial b}$，而输出$\hat{x}'=\mu+(b+\Delta b)-E[\mu+(b+\Delta b)]=\mu+b-E[\mu+b]$，可以看出该层的输出和损失没有任何变化，但随着更新$b$会趋近于无穷。所以在normalization时要考虑优化的过程。
2. 在前向传播进行normalization，反向传播考虑其他样本数据对于当前样本的影响。如果将normalization过程看做一次变化，$\hat{x}=Norm(x, \chi)$。反向传播时考虑其他样本数据对当前样本的影响，需要计算两个导数：$\frac{\partial Norm(x, \chi)}{\partial x}$和$\frac{\partial Norm(x, \chi)}{\partial \chi}$，其中第二个计算的复杂度较大。

为此有两处做了简化：

1. 对每个特征独立的进行normalization。

2. 简单的normalization每一层的输入降低了模型的表现能力，比如norm后通过sigmoid激活函数数据分布在中间线性的区域，所以加入可学习的参数进行scale and shift。

# Method

如果batch size为m，则在前向传播时每个节点都有m个输出，对每个节点的m个输出进行归一化。实现如下图所示，可以分为两步：

 - Standardization：对m个$x$进行标准化得到zero mean unit variance的$\hat{x}$
 - Scale and shift：对$\hat{x}$进行缩放和平移，得到最终的分布$y$，具有新的均值$\beta$和方差$\gamma$

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211006-1.jpg)

针对Batch Normalization Transform，总体可以写为如下公式，所以，无论$x_i$原本的均值和方差是多少，通过Batch Norm后其均值和方差分别变为待学习的$\beta$和$\gamma$。

$y^{(b)}_i=BN(x_i)^{b}=\gamma\cdot(\frac{x_i^{b}-\mu(x_i)}{\sqrt{}\sigma(x_i)^2})+\beta$

---

# Training & Testing

在训练时，$\mu$，$\sigma$是当前mini batch的统计量，随着batch的不同一直在变化。在预测阶段，我们希望模型的输出只和输入相关，所以$\mu$，$\sigma$应该固定，可以采用移动平均来计算$\mu$，$\sigma$。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20211006-2.jpeg)



# Reference

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)