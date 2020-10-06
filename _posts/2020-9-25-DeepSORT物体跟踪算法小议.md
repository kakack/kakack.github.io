---

layout: post
categories: [Algorithm]
tags: [Deep SORT, Tracking]

---

- [Paper](http://arxiv.org/pdf/1602.00763.pdf)
- [Open Source Code](https://github.com/abewley/sort)


# Abstract

`Deep SORT`是多目标跟踪`Multiple Object Tracking(MOT)`中常用到的一种算法，是一个Detection Based Tracking的方法。主要任务是给定一个图像序列，找到图像序列中运动的物体，并将不同帧的运动物体进行识别，也就是给定一个确定准确的id

整个计算架构可以简单地被分为四个主要步骤，当获得一系列待检测的原始视频帧之后：
1. 检测：用如faster-R-CNN、YoloV3、SSD等工具进行目标检测，获得对应目标的检测框，这是其中依赖程度最高的一个步骤；
2. 特征提取：提取被检测目标的表现特征或运动特征
3. 相似度计算：
4. 关联：计算连续数帧画面中被检测物体的关联程度以确定是否同一物体，常用方法为卡尔曼滤波和匈牙利算法。

其中，SORT和DeepSORT的核心就是卡尔曼滤波和匈牙利算法。
- 卡尔曼滤波分为两个过程，预测和更新。该算法将目标的运动状态定义为8个正态分布的向量。`预测`：当目标经过移动，通过上一帧的目标框和速度等参数，预测出当前帧的目标框位置和速度等参数。`更新`：预测值和观测值，两个正态分布的状态进行线性加权，得到目前系统预测的状态。
- 匈牙利算法解决的是一个分配问题，在MOT主要步骤中的计算相似度的，得到了前后两帧的相似度矩阵。匈牙利算法就是通过求解这个相似度矩阵，从而解决前后两帧真正匹配的目标。

而SORT算法中只通过前后两帧IOU来构建相似度矩阵，所以运行速度很快。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20200928-1.JPG)

- - -

# Kalman filter

所谓卡尔曼滤波，简单来说就是在存在一定不确定性噪声的动态系统里，对于目标物体下一步运动进行推测的方法。推测的内容包括之后出现的运动位置以及速度等。

对于卡尔曼滤波而言，目标物体在任意时间点下都有一个状态，和速度、位置相关，且假设变量p位置和v速度都符合随机高斯分布，<a href="https://www.codecogs.com/eqnedit.php?latex=N(\mu,\sigma^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(\mu,\sigma^2)" title="N(\mu,\sigma^2)" /></a>。卡尔曼滤波通过协方差矩阵来描述速度与位置之间的相关性，即矩阵的ij位置上元素的值描述了第i个和第j个变量之间的相关程度。

因此，在完成建模之前还需要补充两个信息，即$k$时刻下，目标物的最佳估计值$和协方差矩阵：

<p><img src="https://i.upmath.me/svg/%5Chat%7Bx%7D_k%3D%5Cbegin%7Bbmatrix%7Dposition%5C%5Cvelocity%5Cend%7Bbmatrix%7D" alt="\hat{x}_k=\begin{bmatrix}position\\velocity\end{bmatrix}" />
<img src="https://i.upmath.me/svg/P_k%3D%5Cbegin%7Bbmatrix%7D%5CSigma_%7Bpp%7D%20%5C%20%5CSigma_%7Bpv%7D%5C%5C%5CSigma_%7Bvp%7D%5C%20%5CSigma_%7Bvv%7D%5Cend%7Bbmatrix%7D" alt="P_k=\begin{bmatrix}\Sigma_{pp} \ \Sigma_{pv}\\\Sigma_{vp}\ \Sigma_{vv}\end{bmatrix}" /></p>

在△t的时间间隔后，假设将位置和速度有如下更新方法：

<a href="https://www.codecogs.com/eqnedit.php?latex=p_k=p_{k-1}&plus;\Delta&space;tv_{k-1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_k=p_{k-1}&plus;\Delta&space;tv_{k-1}" title="p_k=p_{k-1}+\Delta tv_{k-1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=v_k=v_{k-1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?v_k=v_{k-1}" title="v_k=v_{k-1}" /></a>

用矩阵形式表示：

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_k=\begin{bmatrix}&space;1&space;&&space;\Delta{t}\\&space;0&space;&&space;1&space;\end{bmatrix}\hat{x}_{k-1}&space;=F_k\hat{x}_{k-1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{x}_k=\begin{bmatrix}&space;1&space;&&space;\Delta{t}\\&space;0&space;&&space;1&space;\end{bmatrix}\hat{x}_{k-1}&space;=F_k\hat{x}_{k-1}" title="\hat{x}_k=\begin{bmatrix} 1 & \Delta{t}\\ 0 & 1 \end{bmatrix}\hat{x}_{k-1} =F_k\hat{x}_{k-1}" /></a>

根据协方差矩阵性质，可得：

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_k=F_k\hat{x}_{k-1}\\&space;P_k=F_kP_{k-1}F^T_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{x}_k=F_k\hat{x}_{k-1}\\&space;P_k=F_kP_{k-1}F^T_k" title="\hat{x}_k=F_k\hat{x}_{k-1}\\ P_k=F_kP_{k-1}F^T_k" /></a>

引入外部影响：

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_k=F_k\hat{x}_{k-1}&plus;\begin{bmatrix}\frac{\Delta&space;t^2}{2}\\\Delta&space;t\end{bmatrix}a=F_k\hat{x}_{k-1}&plus;B_k\overrightarrow{u}_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{x}_k=F_k\hat{x}_{k-1}&plus;\begin{bmatrix}\frac{\Delta&space;t^2}{2}\\\Delta&space;t\end{bmatrix}a=F_k\hat{x}_{k-1}&plus;B_k\overrightarrow{u}_k" title="\hat{x}_k=F_k\hat{x}_{k-1}+\begin{bmatrix}\frac{\Delta t^2}{2}\\\Delta t\end{bmatrix}a=F_k\hat{x}_{k-1}+B_k\overrightarrow{u}_k" /></a>

其中B为控制矩阵，u为控制变量，如果外部环境异常简单，则可忽略不计。

引入外部不确定性（噪声）Q_k：

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_k=F_k\hat{x}_{k-1}&plus;B_k\overrightarrow{u}_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{x}_k=F_k\hat{x}_{k-1}&plus;B_k\overrightarrow{u}_k" title="\hat{x}_k=F_k\hat{x}_{k-1}+B_k\overrightarrow{u}_k" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=P_k=F_kP_{k-1}F^T_k&plus;Q_k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_k=F_kP_{k-1}F^T_k&plus;Q_k" title="P_k=F_kP_{k-1}F^T_k+Q_k" /></a>

概况而言就是：

- `新的最佳估计`是基于`原最佳估计`和`已知外部影响`校正后得到的预测。
- `新的不确定性`是基于`原不确定性`和`外部环境的不确定性`得到的预测。

- - -

# Reference

- [多目标跟踪：SORT和Deep SORT](https://zhuanlan.zhihu.com/p/59148865)
- [Deep SORT多目标跟踪算法代码解析(上)](https://zhuanlan.zhihu.com/p/133678626)
- [Deep SORT多目标跟踪算法代码解析(下)](https://zhuanlan.zhihu.com/p/133689982)
- [Deep SORT论文阅读总结](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247485748&idx=1&sn=eb0344e1fd47e627e3349e1b0c1b8ada&chksm=9f80b3a2a8f73ab4dd043a6947e66d0f95b2b913cdfcc620cfa5b995958efe1bb1ba23e60100&scene=126&sessionid=1587264986&key=1392818bdbc0aa1829bb274560d74860b77843df4c0179a2cede3a831ed1c279c4603661ecb8b761c481eecb80e5232d46768e615d1e6c664b4b3ff741a8492de87f9fab89805974de8b13329daee020&ascene=1&uin=NTA4OTc5NTky&devicetype=Windows+10+x64&version=62090069&lang=zh_CN&exportkey=AeR8oQO0h9Dr%2FAVfL6g0VGE%3D&pass_ticket=R0d5J%2BVWKbvqy93YqUC%2BtoKE9cFI22uY90G3JYLOU0LtrcYM2WzBJL2OxnAh0vLo)
- [图解卡尔曼滤波](https://zhuanlan.zhihu.com/p/39912633)
- []()
- []()