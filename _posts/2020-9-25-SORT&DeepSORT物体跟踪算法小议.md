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

# SORT

SORT的目标建模是一个八维的模型：

<a href="https://www.codecogs.com/eqnedit.php?latex=X=\begin{bmatrix}\mu&space;&&space;\upsilon&space;&&space;r&space;&&space;h&space;&&space;\dot{x}&space;&&space;\dot{y}&space;&&space;\dot{r}&space;&&space;\dot{h}\end{bmatrix}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X=\begin{bmatrix}\mu&space;&&space;\upsilon&space;&&space;r&space;&&space;h&space;&&space;\dot{x}&space;&&space;\dot{y}&space;&&space;\dot{r}&space;&&space;\dot{h}\end{bmatrix}^T" title="X=\begin{bmatrix}\mu & \upsilon & r & h & \dot{x} & \dot{y} & \dot{r} & \dot{h}\end{bmatrix}^T" /></a>

前三个变量分别表示当前目标在坐标轴的横轴值、纵轴值、BBox的尺寸比例和高，后四个变量为下一帧预测的位置横坐标、纵坐标和BBox尺寸上的相对速度。

SORT使用匈牙利指派算法进行数据关联，使用的cost矩阵为原有目标在当前帧中的预测位置和当前帧目标检测框之间的IOU。当然小于指定IOU阈值的指派结果是无效的。作者发现使用IOU能够解决目标的短时被遮挡问题。这是因为目标被遮挡时，检测到了遮挡物，没有检测到原有目标，假设把遮挡物和原有目标进行了关联。那么在遮挡结束后，因为在相近大小的目标IOU往往较大，因此很快就可以恢复正确的关联。这是建立在遮挡物面积大于目标的基础上的。如果连续`T_lost`帧没有实现已追踪目标预测位置和检测框的IOU匹配，则认为目标消失。实验中设置 `T_lost=1`，原因有二，一是匀速运动假设不合理，二是作者主要关注短时目标追踪。另外，尽早删除已丢失的目标有助于提升追踪效率。

但是在SORT中，目标仅仅根据IOU来进行匹配，必然会导致ID Switch非常频繁。

- - -

# Deep SORT

Deep SORT相比SORT，通过集成表观信息来提升SORT的表现。通过这个扩展，模型能够更好地处理目标被长时间遮挡的情况，将ID switch指标降低了45%。Deep SORT沿用了上述的八维特征来对目标物建模。但是新增了对于目标轨迹的处理和管理，设定了其轨迹的新增、终止、匹配等状态。

在SORT中，我们直接使用匈牙利算法去解决预测的卡尔曼状态和新来的状态之间的关联度，现在我们需要将目标运动和表面特征信息相结合，通过融合这两个相似的测量指标。使用马氏距离来评价预测的卡尔曼状态和新获得的状态：

<a href="https://www.codecogs.com/eqnedit.php?latex=d^{(1)}(i,j)=(d_j-y_j)^TS^{-1}_i(d_j-y_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d^{(1)}(i,j)=(d_j-y_j)^TS^{-1}_i(d_j-y_j)" title="d^{(1)}(i,j)=(d_j-y_j)^TS^{-1}_i(d_j-y_j)" /></a>

表示第j个检测结果和第i条轨迹之间的运动匹配度，其中S_i是轨迹由卡尔曼滤波器预测得到的在当前时刻观测空间的协方差矩阵，y_i是轨迹在当前时刻的预测观测量，d_j是第j个检测结果的状态(u, v, r, h)。

可以用一个指示器的形式描述实际马氏距离跟阈值之间的关系：

<a href="https://www.codecogs.com/eqnedit.php?latex=b_{i,j}^{(1)}=1[d^{(1)}(i,j)\leqslant&space;t^{(1)}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_{i,j}^{(1)}=1[d^{(1)}(i,j)\leqslant&space;t^{(1)}]" title="b_{i,j}^{(1)}=1[d^{(1)}(i,j)\leqslant t^{(1)}]" /></a>

论文中默认阈值t^(1)=9.4877，如果当前马氏距离小于该值则表示匹配成功。

为弥补不确定性较高的时候马氏距离度量失效的问题，会使用第i个轨迹和第j个轨迹之间的最小余弦距离作为第二个度量衡：

<a href="https://www.codecogs.com/eqnedit.php?latex=d^{(2)}(i,j)=min\{1-r_j^Tr_k^{(i)}|r_k^{(i)}\in&space;R_i\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d^{(2)}(i,j)=min\{1-r_j^Tr_k^{(i)}|r_k^{(i)}\in&space;R_i\}" title="d^{(2)}(i,j)=min\{1-r_j^Tr_k^{(i)}|r_k^{(i)}\in R_i\}" /></a>

余弦距离也有一个指示器：

<a href="https://www.codecogs.com/eqnedit.php?latex=b_{i,j}^{(2)}=1[d^{(2)}(i,j)\leqslant&space;t^{(2)}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_{i,j}^{(2)}=1[d^{(2)}(i,j)\leqslant&space;t^{(2)}]" title="b_{i,j}^{(2)}=1[d^{(2)}(i,j)\leqslant t^{(2)}]" /></a>

默认阈值t^(2)=0.2。

综合匹配程度可以将上述二者加权求和：

<a href="https://www.codecogs.com/eqnedit.php?latex=c_{i,j}=\lambda&space;d^{(1)}(i,j)&plus;(1-\lambda)d^{(2)}(i,j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c_{i,j}=\lambda&space;d^{(1)}(i,j)&plus;(1-\lambda)d^{(2)}(i,j)" title="c_{i,j}=\lambda d^{(1)}(i,j)+(1-\lambda)d^{(2)}(i,j)" /></a>

对于每一个轨迹，都计算当前帧距离上次匹配成功的差值。如果产生新的检测结果和现有轨迹无法匹配，则初始化生成一组新的轨迹，新生成的轨迹赋予`不确定态`。而当新轨迹连续三帧都匹配成功，则转换为`确定态`，当某一个轨迹生命周期超过一个最大阈值后认为轨迹离开了画面区域，或连续多帧未匹配成功，则转换为`删除态`，从当前轨迹集合中删除。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20200928-2.png)


级联匹配是Deep SORT区别于SORT的一个核心算法，致力于解决目标被长时间遮挡的情况。为了让当前检测结果匹配上当前时刻较近的轨迹，匹配的时候检测结果优先匹配消失时间较短的轨迹。当目标被长时间遮挡之后，卡尔曼滤波预测结果将增加非常大的不确定性(因为在被遮挡这段时间没有观测对象来调整，所以不确定性会增加)， 状态空间内的可观察性就会大大降低。而当两个轨迹竞争同一个检测结果的时候，消失时间更长的轨迹往往匹配得到的马氏距离更小， 使得检测结果更可能和遮挡时间较长的轨迹相关联，这种情况会破坏一个轨迹的持续性，这也就是SORT中ID Switch太高的原因之一。

因此引入了级联匹配的思路：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20200928-3.png)

- 分配轨迹索引和检测结果索引；
- 计算基于余弦距离的cost matrix；
- 计算基于卡尔曼滤波预测的每个轨迹的平均位置和实际检测获得的BBox之间的平方马氏距离cost matrix；
- 将上述两个cost matrix小于阈值的对应值（匹配成功）设置为无穷大，方便后续计算；
- 将上述两个cost matrix大于阈值的对应值（匹配失败）设置为较大，方便后续删除；
- 使用匈牙利算法对检测结果和轨迹进行匹配，并返回匹配结果；
- 对匹配结果进行筛选，删去外观信息差距过大（即余弦距离过大）的配对；
- 得到初步的匹配结果和未匹配成功的检测结果及轨迹。


- - -

# Reference

- [多目标跟踪：SORT和Deep SORT](https://zhuanlan.zhihu.com/p/59148865)
- [Deep SORT多目标跟踪算法代码解析(上)](https://zhuanlan.zhihu.com/p/133678626)
- [Deep SORT多目标跟踪算法代码解析(下)](https://zhuanlan.zhihu.com/p/133689982)
- [Deep SORT论文阅读总结](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247485748&idx=1&sn=eb0344e1fd47e627e3349e1b0c1b8ada&chksm=9f80b3a2a8f73ab4dd043a6947e66d0f95b2b913cdfcc620cfa5b995958efe1bb1ba23e60100&scene=126&sessionid=1587264986&key=1392818bdbc0aa1829bb274560d74860b77843df4c0179a2cede3a831ed1c279c4603661ecb8b761c481eecb80e5232d46768e615d1e6c664b4b3ff741a8492de87f9fab89805974de8b13329daee020&ascene=1&uin=NTA4OTc5NTky&devicetype=Windows+10+x64&version=62090069&lang=zh_CN&exportkey=AeR8oQO0h9Dr%2FAVfL6g0VGE%3D&pass_ticket=R0d5J%2BVWKbvqy93YqUC%2BtoKE9cFI22uY90G3JYLOU0LtrcYM2WzBJL2OxnAh0vLo)
- [图解卡尔曼滤波](https://zhuanlan.zhihu.com/p/39912633)