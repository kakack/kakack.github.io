---

layout: post
categories: [Computer Graphic]
tags: [Computer Graphic, OpenCV, Face Detection]

---

1. 使用Haar-like特征做检测。
2. 使用积分图（Integral Image）对Haar-like特征求值进行加速。
3. 使用AdaBoost算法训练区分人脸和非人脸的强分类器。
4. 使用筛选式级联把强分类器级联到一起，提高准确率。

简单说来，PAC学习模型不要求你每次都正确，只要能在多项式个样本和多项式时间内得到满足需求的正确率，就算是一个成功的学习。“学习"是模式明显清晰或模式不存在时仍能获取知识的一种“过程”，并给出了一个从计算角度来获得这种“过程"的方法，这种方法包括：

1. 适当信息收集机制的选择；
2. 学习的协定；
3. 对能在合理步骤内完成学习的概念的分类。

强学习和弱学习：

- 弱学习：一个学习算法对一组概念的识别率只比随机识别好一点
- 强学习：一个学习算法对一组概率的识别率很高

- - -

## 人脸识别的应用步骤：

- 人脸检测(Face Detection)。检测到人脸所在的区域。并进行一系列的矫正。
- 人脸校准(Face Alignment)。人脸校准指的是在图片中寻找到鼻子、眼睛、嘴巴之类的位置。
- 信息识别(Info Recognition)。进行性别、年龄等信息的分析和识别。

基于知识的人脸检测方法

- 模板匹配
- 人脸特征
- 形状与边缘
- 纹理特性
- 颜色特征

基于统计的人脸检测方法

- 主成分分析与特征脸
- 神经网络方法
- 支持向量机
- 隐马尔可夫模型
- AdaBoost算法

- - -

## 人脸检测Viola-Jones算法

- Haar-like特征
- AdaBoost分类器
- Cascade级联分类器

##### Haar-like特征：

![](http://ope2etmx1.bkt.clouddn.com/23115756-851f494ea9994b6e90006949a2150c5f.jpg)

一个矩形哈尔特征可以定义为矩形中几个区域的像素和的差值，可以具有任意的位置和尺寸。这种特质也被称为2矩形特征（2-rectangle feature）。 维奥拉和琼斯也定义了3矩形特征和4矩形特征。这个值表明了图像的特定区域的某些特性。每一个特征可以描述图像上特定特性的存在或不存在，比如边缘或者纹理的变化。

定义该模板的特征值为模板矩阵内白色矩形像素和减去黑色矩形像素和。

```
haar-like feature = sum(white) - sum(black)
# 上图A、B、D
haar-like feature = sum(white) - 2*sum(black)
# 上图C，为了使得两种矩形区域内像素数目一致
```

其他一些特征模板

![](http://ope2etmx1.bkt.clouddn.com/206_761_537.jpg)

矩阵特征值根据*特征模板的大小、位置和模板类型*的不同而不同，因此是关于这三个因素的函数。

同时定义了积分图（Integral Image），是一张与原图像大小完全相同的图片，每个点记录了该点对应位置左上方所有像素的和。

<a href="https://www.codecogs.com/eqnedit.php?latex=B(x,&space;y)&space;=&space;\sum_{i\leq&space;x,&space;j\leq&space;y}&space;A(i,&space;j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B(x,&space;y)&space;=&space;\sum_{i\leq&space;x,&space;j\leq&space;y}&space;A(i,&space;j)" title="B(x, y) = \sum_{i\leq x, j\leq y} A(i, j)" /></a>

利用积分图可以只查询4次就求得一个2矩阵特征的值：

![](http://ope2etmx1.bkt.clouddn.com/211_fb0_159.jpg)

如图所示白色部分计算结果即`B(5)+B(1)-B(2)-B(6)`，黑色部分计算结果即`B(4)+B(2)-B(3)-B(5)`，二者的差即所示矩阵的特征值。

##### AdaBoost算法

AdaBoost的核心思想就是从多个弱分类器中结合，生成一个强分类器。数学书表达就是：

<a><img src="https://latex.codecogs.com/gif.latex?F(x)&space;=&space;\sum_{i&space;=&space;1}^{N}&space;\alpha_{i}&space;f_{i}(x)" title="F(x) = \sum_{i = 1}^{N} \alpha_{i} f_{i}(x)" /></a>

其中F是强分类器，f是各个弱分类器，x是特征向量，α是各个弱分类器对应的权重，N是弱分类器的个数。

其中弱分类器定义为：

<a><img src="https://latex.codecogs.com/gif.latex?h_{j}(x)&space;=&space;\begin{cases}&space;&&space;\text{&space;1&space;if&space;}&space;p_{j}f_{j}(x)&space;<&space;p_{j}\Theta&space;_{j}&space;\\&space;&&space;\text{&space;0&space;if&space;}&space;otherwise&space;\end{cases}" title="h_{j}(x) = \begin{cases} & \text{ 1 if } p_{j}f_{j}(x) < p_{j}\Theta _{j} \\ & \text{ 0 if } otherwise \end{cases}" /></a>

其中<a><img src="https://latex.codecogs.com/gif.latex?f_{j}(x)" /></a>表示一个输入窗口x，通过这个函数提取特征值，然后通过一个阈值θ判定是否是目标物体。<a href="https://www.codecogs.com/eqnedit.php?latex=p_{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{j}" title="p_{j}" /></a>是为了控制不等式左右的符号。

假设样本图像为<a><img src="https://latex.codecogs.com/gif.latex?(x_{i},&space;y_{i})"/></a>，共m个，初始化权重<img src="https://latex.codecogs.com/gif.latex?w_{1, i} = \frac{1}{2m}"/>

For t=1, 2, 3...T:

1. 归一化权重<img src="https://latex.codecogs.com/gif.latex?w_{t,i}=\frac{w_{t,i}}{\sum_{j=1}^{n}w_{t,j}}"/>
2. 对于每个特征，训练并记录一个分类器，记为<img src="https://latex.codecogs.com/gif.latex?h_{i}"/>，每个分类器都只使用一个特征进行训练，而对于该特征的误差<img src="https://latex.codecogs.com/gif.latex?\epsilon_{j} "/>可以衡量：<img src="https://latex.codecogs.com/gif.latex?\epsilon_{j} = \sum_{i} w_{i}\left | h_{j}(x_{i})-y_{i} \right |"/>
3. 选择拥有最低误差的分类器记为<img src="https://latex.codecogs.com/gif.latex?h_{t}"/>
4. 更新权重<img src="https://latex.codecogs.com/gif.latex?w_{t+1, i}=w_{t, i}\beta_{t}^{1-e_{i}}"/>，如果分类正确<img src="https://latex.codecogs.com/gif.latex?e_{i} = 1"/>，错误<img src="https://latex.codecogs.com/gif.latex?e_{i} = 0"/>

End for

最终得到强分类器：

<img src="https://latex.codecogs.com/gif.latex?h(x)=\begin{cases}&space;&&space;\text{&space;1&space;if&space;}&space;\sum_{t=1}^{T}\alpha_{t}h_{t}&space;\geq&space;\frac{1}{2}\sum_{t=1}^{T}\alpha_{t}&space;\\&space;&&space;\text{&space;0&space;}&space;otherwise&space;\end{cases}" title="h(x)=\begin{cases} & \text{ 1 if } \sum_{t=1}^{T}\alpha_{t}h_{t} \geq \frac{1}{2}\sum_{t=1}^{T}\alpha_{t} \\ & \text{ 0 } otherwise \end{cases}" />

其中<img src="https://latex.codecogs.com/gif.latex?\alpha_{t}=log\frac{1}{\beta_{t}}"/>

![](http://ope2etmx1.bkt.clouddn.com/573f177219528.gif)

##### Cascade级联分类器

就是讲几个通过AdaBoost方法得到的强分类器进行排序，简单的在左复杂的在右，因为认为一般照片中人脸所占部分较小，可以较为大胆地去除很大一部分非人脸的内容

![](http://ope2etmx1.bkt.clouddn.com/20150805233022701.jpg)
 
 - - -
 
 ## OpenCV-Python中的例子
 
 ```Python
import cv2

imagepath ='./catface2.jpg'
# 设定待识别的图像在文件夹中的位置

image = cv2.imread(imagepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 图片读入，转化为灰度图

face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalcatface_extended.xml')
# haarcascade_frontalface_default.xml是OpenCV自带的根据haar-like特征值已经训练好的模型，可以在opencv文件夹下./share/OpenCV/haarcascades文件夹中找到，还有其他不同的多种识别模型

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (5,5),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))
# 打印出识别信息

for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
# 根据识别结果画长方形显示面部

cv2.imshow("Find Faces!",image)
cv2.waitKey(0)
 ```
 

