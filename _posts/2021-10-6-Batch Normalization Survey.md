---

layout: post
categories: [Algorithm]
tags: [Deep Learning, Batch Normalization]

---

# What is Batch Normalization？

**Batch Normalization**（以下用BN简称代替）是为了解决Deep Learning中*Internal Covariate Shift（内部协变量移位）*问题，而针对一个batch中的数据进行归一标准化的方法。作用是可以使用更加flexible的学习率learning rate，得到更快的学习速率，同时不会过于依赖模型的初始化initialization。其中所谓的Internal Covariate Shift是指在deep neural network当中，随着层数的加深，每一层layer都关联着其上下层，其input实上一层layer的output，而其output又是下一层layer的input，而在随机梯度下降学习过程中，随着每层layer参数的更新，其对应的input/output分布也在随时起着微小的变化，这种变化会随着层数深度的加深而逐步积累，从而使得整个学习过程难度加大。

# Introduce BN

BN的整体思路其实是一种*whiten（百化）*操作，即将输入的数据分布转换到均值$\mu=0$，方差$\sigma=1$的正态分布，能有效加快整个network的收敛，即用较少的学习步骤获得相同的推理精度。直观上的现象就是缓解梯度消失，支持更大的学习率。因为大学习率往往导致反向时梯度困在局部最小值处，BN可以避免神经网络层中很小的参数变动在层数加深的过程中会积聚造成很大的影响。

---

# Reference

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)