---

layout: post
categories: [Algorithm]
tags: [Deep Learning, Batch Normalization]

---

# What is Batch Normalization？

**Batch Normalization**（以下用BN简称代替）是为了解决Deep Learning中*Internal Covariate Shift（内部协变量移位）*问题，而针对一个batch中的数据进行归一标准化的方法。作用是可以使用更加flexible的学习率learning rate，得到更快的学习速率，同时不会过于依赖模型的初始化initialization。其中所谓的Internal Covariate Shift是指在deep neural network当中，随着层数的加深，每一层layer都关联着其上下层，其input实上一层layer的output，而其output又是下一层layer的input，而在随机梯度下降学习过程中，随着每层layer参数的更新，其对应的input/output分布也在随时起着微小的变化，这种变化会随着层数深度的加深而逐步积累，从而使得整个学习过程难度加大。


$$(a_1 + b)x^2 = c$$
$$(W_1-W_2)x+b_1-b_2=0$$

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$

This is inline $$\sum_{i=1}^n x_ie_i$$

The following is a math block:

$$\sum_{i=1}^n x_ie_i$$

But next comes a paragraph with an inline math statement:

\$$\sum_{i=1}^n x_ie_i$$


---

# Reference

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)