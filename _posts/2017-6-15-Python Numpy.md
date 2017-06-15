---

layout: post
categories: [Python]
tags: [Python, Numpy]

---

# Numpy Module

`Numpy`的核心对象是均匀的齐次多维数组

`Numpy`的维度dimension叫做*axes*，其中*axes*的数量叫做*rank*

称为`ndarray`，也可以叫做`array`，但是与标准Python库中的`array`不同。

`ndarray`的属性有以下：
  - ndarray.ndim：array的维度`axes`的数量，即`rank`
  - ndarray.shape：array的维度，返回一个tuple，表示array中每个维度的大小，比如一个n行m列的矩阵，shape值为(n, m) ，因此shape值的长度就是*rank*，也就是`ndarray.ndim`
  - ndarray.size：array中所有元素的个数，等于`shape`中各个值的乘积
  - ndarray.dtype：array中元素的类型
  - ndarray.itemsize：array中每个元素占的byte数
  - ndarray.data：保存着array真实元素的buffer，通常不需要
  
  - - -

## Array创建

可以通过一个通常的Python list或者tuple创建，如：

```python
>>> a = np.array([1,2,3])
>>> a
array([1, 2, 3])
```

注，需要以list作为单个参数，传入，而不是传入若干个数字参数：`a = np.array([1,2,3]) # wrong`

也可以用`zeros`或者`ones`创建全是0或者全是1的array：

```python

>>> np.zeros( (3,4) )
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
>>> np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
>>> np.empty( (2,3) )                                 # uninitialized, output may vary
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
```

或者通过`arange`函数创建序列的array

```python
>>> a = np.arange(6)                         # 1d array
>>> print(a)
[0 1 2 3 4 5]
>>>
>>> b = np.arange(12).reshape(4,3)           # 2d array
>>> print(b)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
>>>
>>> c = np.arange(24).reshape(2,3,4)         # 3d array
>>> print(c)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

- - -

## 基本操作

```Python
>>> a = np.array( [20,30,40,50] )
>>> b = np.arange( 4 )
>>> b
array([0, 1, 2, 3])
>>> c = a-b
>>> c
array([20, 29, 38, 47])
>>> b**2
array([0, 1, 4, 9])
>>> 10*np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
>>> a<35
array([ True, True, False, False], dtype=bool)

```

其中`+=`和`*=`操作都是在原有基础上修改array对象，而不是创建一个新的对象

- - -

## 通用函数

提供`sin`, `cos`, `exp`等数学函数，称为`universal functions（ufunc）`

```python
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> np.sqrt(B)
array([ 0.        ,  1.        ,  1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([ 2.,  0.,  6.])
```

## 索引、分片和迭代

对于一维的array来说，可以被索引、分片和迭代，就像普通的Python list

```python
>>> a = np.arange(10)**3
>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
>>> a[2]
8
>>> a[2:5]
array([ 8, 27, 64])
>>> a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
>>> a
array([-1000,     1, -1000,    27, -1000,   125,   216,   343,   512,   729])
>>> a[ : :-1]                                 # reversed a
array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1, -1000])
>>> for i in a:
...     print(i**(1/3.))
...
nan
1.0
nan
3.0
nan
5.0
6.0
7.0
8.0
9.0
```

- - -

## 修改array的shape

假设有一个（3，4）的array，如何修改其shape，有如下办法：

```python
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.shape
(3, 4)

# 扁平化 

>>> a.ravel()  # returns the array, flattened
array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])

# reshape()命令

>>> a.reshape(6,2)  # returns the array with a modified shape
array([[ 2.,  8.],
       [ 0.,  6.],
       [ 4.,  5.],
       [ 1.,  1.],
       [ 8.,  9.],
       [ 3.,  6.]])

>>> a.T  # returns the array, transposed
array([[ 2.,  4.,  8.],
       [ 8.,  5.,  9.],
       [ 0.,  1.,  3.],
       [ 6.,  1.,  6.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
```

- - -

## 将不同的Array粘到一起

不同维度的array可以被粘在一起：`vstack()`命令

```python
>>> c
array([False, False,  True,  True], dtype=bool)
>>> d
array([False,  True,  True,  True], dtype=bool)
>>> np.vstack((c,d))
array([[False, False,  True,  True],
       [False,  True,  True,  True]], dtype=bool)
>>> e = np.vstack((c,d))
>>> e.ravel()
array([False, False,  True,  True, False,  True,  True,  True], dtype=bool)
# 这个操作可以用到cifar10的eval里
```

## 将一个Array拆分成若干个小arrays

可以指定维度来拆分成特定个数的arrays

```Python
>>> a = np.floor(10*np.random.random((2,12)))
>>> a
array([[ 6.,  1.,  0.,  1.,  0.,  6.,  4.,  0.,  7.,  6.,  6.,  2.],
       [ 9.,  1.,  8.,  5.,  0.,  1.,  6.,  1.,  3.,  4.,  4.,  8.]])
# 生成一个（2，12）的array
>>> np.hsplit(a, 3)
[array([[ 6.,  1.,  0.,  1.],
       [ 9.,  1.,  8.,  5.]]), array([[ 0.,  6.,  4.,  0.],
       [ 0.,  1.,  6.,  1.]]), array([[ 7.,  6.,  6.,  2.],
       [ 3.,  4.,  4.,  8.]])]
# 生成三个(2, 4)的array
>>> np.hsplit(a, (3, 4))
[array([[ 6.,  1.,  0.],
       [ 9.,  1.,  8.]]), array([[ 1.],
       [ 5.]]), array([[ 0.,  6.,  4.,  0.,  7.,  6.,  6.,  2.],
       [ 0.,  1.,  6.,  1.,  3.,  4.,  4.,  8.]])]
```

- - -

## Copy和Views

当使用和操作时，有时会将array复制到一个新的array里，但有时仅仅是做了引用：

```Python
# 这里b和a是同一个对象的不同名称而已

>>> a = np.arange(12)
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
>>> b.shape = 3,4    # changes the shape of a
>>> a.shape
(3, 4)
```
`View`方法创造一个新的对象，其中数据一致

```python
>>> c = a.view()
>>> c is a
False
>>> c.base is a                        # c is a view of the data owned by a
True
>>> c.flags.owndata
False
>>>
>>> c.shape = 2,6                      # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0,4] = 1234                      # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```
将一个array做切片操作
```python
>>> s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
>>> s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

Deep Copy

```Python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```