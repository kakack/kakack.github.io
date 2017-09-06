---

layout: post
categories: [Deep Learning]
tags: [Deep Learning, Machine Learning]

---

# 通过神经网络mindset实现简单的Logistic Regression

## 1 - Package

- [numpy](www.numpy.org) Python科学计算的基础包
- [h5py](http://www.h5py.org) 与存放在H5文件中的数据集进行科学交互的包
- [matplotlib](http://matplotlib.org) Python绘图包
- [PIL](http://www.pythonware.com/products/pil/) 和 [scipy](https://www.scipy.org/) 用来在最后测试自定义的图片

```Python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline
```

---

## 2 - 问题数据集概述

**问题陈述：**给定衣柜数据集（“data.h5”）包含：
		
- 一个包含了`m_train`图片的训练集，分别被标注了是猫(y=1)或者不是猫(y=0)
- 一个包含了`m_test`图片的测试集，也分别被标注了是猫或者不是猫
- 每个图片的shape都是(`num_px`, `num_px`, `3`)，其中3表示图像有3个频道（RGB），因此每个图像都是一个正方形（`height = num_px`, `width = num_px`）

```Python
# Loading the data (cat/non-cat)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

```

载入图片，在变量后加后缀`_orig`表示这是原数据，仍需要进一步处理，其中`train_set_x_orig`和`train_set_x_orig`每一行都代表了一张图片，可以通过下面的例子查看当前的图片：

```Python
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
```

定义变量：

- m_train：训练集的个数
- m_test：测试集的个数
- num_px：训练图像的长和宽

```Python
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Number of training examples: m_train = 209
# Number of testing examples: m_test = 50
# Height/Width of each image: num_px = 64
# Each image is of size: (64, 64, 3)
# train_set_x shape: (209, 64, 64, 3)
# train_set_y shape: (1, 209)
# test_set_x shape: (50, 64, 64, 3)
# test_set_y shape: (1, 50)
```

为了方便起见，将图片reshape到一个新的格式：（num_px*num_px*3, 1），这样之后我们的训练集和测试集都是一个扁平化的numpy-array，每一列都代表了一个图像，其中总共m_train(或m_test)列。

```Python
# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# train_set_x_flatten shape: (12288, 209)
# train_set_y shape: (1, 209)
# test_set_x_flatten shape: (12288, 50)
# test_set_y shape: (1, 50)
# sanity check after reshaping: [17 31 56 22 33]

```

为了表示一个彩色图片，，每一个像素都是一个三个值组成的向量，每个值都位于区间[0, 255]。一种常见的机器学习预处理手段是将数据集中心化和标准化，这意味着在每个例子上都减去整个numpy array的平均值，然后每个例子都除以整个numpy array的标准偏差，而对于图片数据集，就可以简单地只是将每一行除以255（一个像素通道的最大值）。

```Python
# Standardize our dataset.
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```

一般的预处理步骤：

- 找出问题数据的维度和shape（m_train, m_test, num_px, ...）
- 对数据集做reshape，使得其向量化(num_px * num_px * 3, 1)
- [标准化数据](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)

---

## 3 - 学习算法基本架构

![](http://ope2etmx1.bkt.clouddn.com/LogReg_kiank.png)

数学表达式：

对于一个例子<img src="http://latex.codecogs.com/gif.latex?x^{(i)}"/>：

- <img src="http://latex.codecogs.com/gif.latex?z^{(i)} = w^T x^{(i)} + b"/>
- <img src="http://latex.codecogs.com/gif.latex?\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})"/>
- <img src="http://latex.codecogs.com/gif.latex?\mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})"/>

其中损失函数是所有训练例子的总和：

<img src="http://latex.codecogs.com/gif.latex?J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})"/>

---

## 4 - 建立算法的各个部分

建立神经网络的主要步骤：

1. 定义模型类型和结构（例如输入的特征数量等）
2. 初始化模型参数
3. 循环：
	- 计算当前的损失（forward propagation）
	- 计算当前的梯度（backward propagation）
	- 更新参数（gradient descent）

### 4.1 Helper Function

计算<img src="http://latex.codecogs.com/gif.latex?sigmoid( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}"/>

```Python
# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1./(1.+np.exp(-z))
    
    return s
```

### 4.2 初始化参数：

```Python
# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

```


### 4.3 Forward & Backward propagation

Forward Propagation：

- 得到X
- 计算<img src="http://latex.codecogs.com/gif.latex?A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})"/>
- 计算损失函数<img src="http://latex.codecogs.com/gif.latex?J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})"/>

其中：

- <img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T"/>
- <img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})"/>

```Python
# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)                                      # compute activation
    cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))/(-m)                                # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X, (A - Y).T)/m
    db = np.sum(A - Y)/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```

### 4.4 优化

更新各个参数来实现梯度下降

```Python

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

```

预测函数，分两步进行预测：

- 计算<img src="http://latex.codecogs.com/gif.latex?\hat{Y} = A = \sigma(w^T X + b)"/>
- 将结果转化为0（如果activation<=0.5）或者1（如果activation>0.5）

```Python
# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = np.dot(w.T, X)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if (A[0, i] > 0.5):
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


```

---

## 5 - 将算法的各个部分汇总

- `Y_prediction`作为测试集的预测结果
- `Y_prediction_train`作为训练集的预测结果
- `w`, `costs`, `grads`作为`optimize()`的输出


```Python

# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
        
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

```

查看结果：

```Python
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# train accuracy: 98.08612440191388 %
# test accuracy: 70.0 %
```

这个模型在训练集上表现很好，但是在测试集上表现一般，所以很显然存在过拟合的问题，之后可以通过正则化等方法来解决过拟合问题。

---

## 6 - 后续分析

通过控制学习率Learning Rate的大小来查看学习曲线的变化情况

```Python

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

```

- 不同的学习率能得到不同的损失值，因此也会有不同的预测结果
- 如果学习率太大，损失值会过于陡峭地上下震荡，甚至会发生分歧
- 一个很低的学习率并不代表一定会得到一个很好的模型，因为需要看看是否出现过拟合了，这在训练集预测结果大大好于测试集预测结果时发生
- 在深度学习中，我们常常建议：
	-  选择能最小化损失函数的学习率
	-  如果模型过拟合了，使用其他手段来解决