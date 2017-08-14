---

layout: post
categories: [Neural Network]
tags: [Neural Network, TensorFlow]

---

## Introduce

TensorFlow是用于数值计算的开源软件库，是其他机器学习框架中最小的。 它最初由Google Brain Team的研究人员和工程师开发，目的是鼓励对深层架构的研究。 然而，TensorFlow环境提供了一套适用于数字编程领域的大量工具。 计算是在数据流图的概念下进行的。 图中的节点表示数学运算，而图形边缘则代表张量（多维数据阵列）。 软件包的核心是用C ++编写，但提供了一个很好的文档Python API。 主要特征是其象征性的方法，其允许对正向模型的一般定义，使得相应衍生物的计算完全与环境本身相关。


**The Data Flow Graph**: 为了利用多核CPU，GPU甚至GPU集群的并行计算能力，数值计算的动态被认为是有向图，其中每个节点表示数学运算，边界描述了输入/输出关系节点。

**Tensor**: 一个流过Data Flow Graph的n维数组类型

**Variable**：用于表示参数的符号对象。 他们在符号级别被利用来计算
衍生结果和中间变量，但通常必须在Session中显式初始化。

**Optimizer**：它是提供从损失函数计算梯度的方法并通过所有变量应用反向传播的组件。 TensorFlow中提供了一个集合来实现经典优化算法。

**Session**：一个Graph必须在Session中启动，它将graph安置到CPU或者GPU上并为其运行计算提供方法。

- - -

## 用TensorFlow解决XOR问题


导入TensorFlow包，创建用于训练的X和Y值
```Python
import tensorflow as tf

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
```

设置占位符
```
x_ = tf.placeholder(tf.float32, shape=[4, 2])
y_ = tf.placeholder(tf.float32, shape=[4, 1])

HU = 3
```

初始化权重向量，一开始以随机值为值。其中`tf.nn.sigmoid`为一种激活函数，`y = 1 / (1 + exp(-x))`, 用处是与其它层的输出联合使用可生成特征图，用于对某些运算的结果进行平滑或微分，如tf.nn.relu。激活函数是用来加入非线性因素的，因为线性模型的表达能力不够。
```
W1 = tf.Variable(tf.random_uniform([2, HU], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([HU]))
O = tf.nn.sigmoid(tf.matmul(x_, W1) + b1)

W2 = tf.Variable(tf.random_uniform([HU, 1], -1.0, 1.0))
b2 = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(O, W2) + b2)

# 2-layer ann

```

计算损失值，训练就使用梯度下降的办法`GradientDescentOptimizer`

```Python
cost = tf.reduce_sum(tf.square(y_ - y), reduction_indices=[0])
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
```
设置tf运行的Session，并将占位符初始化
```Python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

设置运行运行步长

```Python
Ecoches = 5000
for i in range(Ecoches):
    sess.run(train_step, feed_dict={x_ : X, y_ :Y})
    if i%500 == 0:
        print ('Epoch ', i)
        print ('Cost ', sess.run(cost, feed_dict={x_: X, y_: Y}))

```

计算预测值与实际值之间的准确率
```Python
correcct_prediction = abs(y_ - y) < 0.5
cast = tf.cast(correcct_prediction, "float")
accuracy = tf.reduce_mean(cast)
```

输出结果
```Python
yy, aa = sess.run([y, accuracy], feed_dict={x_: X, y_: Y})

print "Output: ", yy
print "Accuracy: ", aa

```

```
Output:  [[ 0.10211191]
 [ 0.91052949]
 [ 0.88228792]
 [ 0.08940154]]
Accuracy:  1.0

```