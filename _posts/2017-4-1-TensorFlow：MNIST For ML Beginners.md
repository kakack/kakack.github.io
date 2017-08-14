---

layout: post
categories: [Neutral Network]
tags: [TensorFlow, Neutral Network]

---

- 创建一个softmax回归函数，该函数是通过查看图像中的每个像素从而识别MNIST数字的模型
- 通过查看上千个例子，来使用TensorFlow训练模型识别数字
- 用测试数据检验模型的准确性
- 创建、训练和测试一个多层卷积神经网络来提高结果

---

# 载入MNIST DATA

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```
其中`mnist`是一个轻量级的类，以`Numpy arrays`的形式保存了训练、验证和测试集

下载下来的数据集被分成两部分：60000行的训练数据集`（mnist.train）`和10000行的测试数据集`（mnist.test）`。

其中每个数据单元都由两部分组成，一张手写数字图片和一个对应标签，将图片设为`xs`，标签设为`ys`

可以用以下方法画9张图像，然后在下面显示预测类别和真实类别

```Python
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
```

然后绘制几张看看效果

```python
# Get the first images from the test-set.
images = data.test.images[0:9]
# Get the true classes for those images.
cls_true = data.test.labels[0:9]
# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)
```

![img](https://pic1.zhimg.com/v2-842b6427471fa494db8748943d5ee91c_b.png)





---

# 启动TensorFlow InteractiveSession

TensorFlow依靠高效的C ++后端来进行计算，与此后端的连接称为会话Session， TensorFlow程序的常见用法是首先创建一个图形Graph，然后在会话中启动它。

时常会用如`Numpy`这样的python包来进行复杂的操作和计算，比如矩阵的计算，在GPU和分布式计算方式中，会有很高的cost用在数据转移传输上。其中Python代码的角色是创建一个外部计算的Graph，然后指定哪一部分的Graph需要被运行。

---

# 创建Softmax Regression Model

### Placeholders

开始创建图像输入和输出的节点：

```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```
这里的x和y_不是特定的变量，而是一个占位符placeholder，之后会询问TernsorFlow计算得到这个输入。

作为输入的图像`x`将会是一个由浮点数组成的2d的tensor，我们把它关联到一个`[None, 784]`的形状，其中`784`是单个平坦化28×28像素MNIST图像的维度，`None`指明的是第一个维度，对应于`batch`大小，可以是任意大小。目标输出类`y_`同样是由2d的tensor组成，其中每一行都是一个one-hot的10维向量，表示对应的MNIST图像属于哪个数字类（0到9）。

### 变量 Variable

我们现在定义我们的模型的权重W和偏差b。我们可以想象，像其他输入一样处理这些信息，但是TensorFlow有一个更好的处理方式：变量Variable。变量是居住在TensorFlow计算图中的值。它可以被计算使用甚至修改。在机器学习应用中，一般通常将模型参数设为变量。

```pytthon
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

我们通过调用`tf.Variable`方法传递每个参数的初始值。在这种情况下，我们都将W和b初始化为充满零的tensor。W是784x10矩阵（因为我们有784个输入特征和10个输出），而b是10维向量（因为我们有10个类）。

在会话session中可以使用变量variable之前，所有变量需要用session来初始化。在此可以一次就对所有已初始化的值关联到变量上。

```python
sess.run(tf.global_variables_initializer())
```

### 预测类和损失函数

开始使用回归模型，代码只有一行。我们将向量化的输入图像`x`乘以权重矩阵`W`，加上偏移量`b`。

```python
y = tf.matmul(x,W) + b
```

我们可以很容易地指定一个损失函数。损失Loss表明模型在一个例子上的预测如果出现错误会有多糟糕;我们尽量在例子的训练中减少这样的损失。在这里，我们的损失函数是应用于模型预测的目标和softmax激活函数之间的交叉熵。 在初学者教程中，我们使用了稳定的公式：

```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes, and tf.reduce_mean takes the average over these sums.

注意`tf.nn.softmax_cross_entropy_with_logits`内部将softmax应用于模型的非规则化模型的预测和所有类别的总和，`tf.reduce_mean`将所得的和取平均。

---

# 训练模型

在定义好模型和训练损失函数之后，可以通过TensorFlow进行训练，因为TensorFlow知道整个计算的图，所以可以使用自动差分来找到每个变量的损失梯度。TensorFlow用很多内建的优化算法，比如可以使用最陡峭梯度下降，将每次的下降step设定为0.5，来降低交叉熵：

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

TensorFlow实际做的事情是在计算图中新加了一个操作，这些操作包括计算梯度、更新steps后计算参数、根据参数更新steps。

返回的操作`train_step`在运行时将对参数应用梯度下降更新。因此，训练模型可以通过重复运行train_step来实现。

```python
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

我们在每个训练迭代中加载100个训练样例。然后我们运行`train_step`操作，使用`feed_dict`替换训练样例的占位符tensor`x`和`y_`。

### 评估模型

整个模型表现如何？

首先，我们将弄清楚我们是否预测了正确的标签。`tf.argmax`是一个非常有用的功能，它可以给出沿某个轴的tensor中最高条目的索引。例如，`tf.argmax（y，1）`是我们的模型认为对每个输入最有可能的标签，而`tf.argmax（y_，1）`是真实标签。我们可以使用`tf.equal`比较两个值来检查我们的预测是否符合真相。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

这给了我们一个布尔的列表。为了确定哪个部分是正确的，我们转换为浮点数，然后取平均值。例如，[True，False，True，True]将变为[1,0,1,1]，这将变为0.75。

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

最终可以评估得到测试数据的准确性。大概有92%的准确性。

```python
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```
---

# 构建多层卷积网络

在MNIST上获得92％的准确性是不好的，几乎是尴尬的不能接受的结果。在本节中，我们将修复这个问题，从一个非常简单的模型跳到中等程度：一个小的卷积神经网络。 这将使我们达到约99.2％的准确度。

### 权重初始化

要创建这个模型，我们将需要创建很多权重`weight`和偏差`bias`。通常应该用少量的噪声来初始化重量以进行对称断裂`symmetry breaking`，并且防止0梯度。由于我们使用`ReLU`神经元，所以用一个轻微的初始化它们初始偏移量以避免“死神经元”也是一个很好的尝试。而不是在构建模型时反复执行，我们创建两个方便的函数：

```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

### 卷积和集合

TensorFlow还为卷积和集合操作提供了很大的灵活性。我们如何处理边界？我们的步幅是多少？在这个例子中，我们总是选择vanilla版本。我们的卷积使用步长为1，零填充，使输出与输入的大小尺寸相同。为了使代码更清洁，我们还将这些操作抽象为函数。

```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

### 第一层卷积层

我们现在可以实现我们的第一层。 它将由卷积组成，其次是最大合并。 卷积将为每个5x5的patch计算32个功能。其权重tensor将具有[5,5,1,32]的形状。前两个维度是patch大小，下一个是输入通道的数量，最后一个是输出通道的数量。我们还将拥有一个带有每个输出通道分量的偏置矢量。

```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

为了应用这个层，我们首先将`x`重新变化成一个四维的tensor，其中第二和第三个维度对应的是宽和高，最后一个维度对应的是颜色通道的数量。

```python
x_image = tf.reshape(x, [-1,28,28,1])
```

我们之后用权重tensor卷积`x_image`，加上偏移量，应用`ReLU`函数，最终将集合最大化。其中`max_pool_2x2`方法可以将图像大小减到14x14。

```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

### 第二层卷积层

为了构建一个深层次的网络，我们堆叠若干层这种类型的层。 第二层将为每个5x5patch提供64个特征。

```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

### 密集层

既然现在图像尺寸已经缩小到7x7，我们添加了一个具有1024个神经元的完全连接的图层，以便对整个图像进行处理。我们从汇集层将tensor重塑成一批向量，乘以权重矩阵，添加偏倚并应用ReLU。

```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

### [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

为了减少过度拟合，我们将在读出层之前应用压差dropout的办法。我们创建一个占位符，以便在dropout时保存神经元的输出。这样可以让我们在训练过程中开启dropout，并在测试过程中将其关闭。TensorFlow的`tf.nn.dropout op`自动处理缩放神经元输出，并且掩盖它们，所以dropout只是在没有任何额外的缩放时才工作。

```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

### 读出层

最后我们添加一层，就像是之前的softmax回归层一样。

```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

### 训练和评估模型

最终模型表现如何？为了训练和评估它，我们将使用与上面简单的SoftMax网络层几乎相同的代码。

但是区别在于：

- 我们将用更复杂的ADAM优化器替换最陡峭的梯度下降优化器。
- 我们将在`feed_dict`中添加附加参数`keep_prob`来控制dropout率。
- 我们将在训练过程中每100次添加日志记录。

```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

运行此代码后的最终测试集精度应为大约99.2％。