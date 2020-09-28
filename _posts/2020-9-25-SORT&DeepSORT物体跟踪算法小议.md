---

layout: post
categories: [Algorithm]
tags: [Deep SORT, Tracking]

---

- [Paper](http://arxiv.org/pdf/1602.00763.pdf)
- [Open Source Code](https://github.com/abewley/sort)


# Abstract

`Deep SORT`是多目标跟踪`Multiple Object Tracking(MOT)`中常用到的一种算法，是一个Detection Based Tracking的方法。主要任务是给定一个图像序列，找到图像序列中运动的物体，并将不同帧的运动物体进行识别，也就是给定一个确定准确的id

整个计算架构可以简单地被分为三个主要步骤，当获得一系列待检测的原始视频帧之后：
1. 检测：用如faster-R-CNN、YoloV3、SSD等工具进行目标检测，获得对应目标的检测框，这是其中依赖程度最高的一个步骤；
2. 特征提取和reid：提取被检测目标的表现特征或运动特征并进行reid的操作，常用的reid模型如wide residual network
3. 关联：计算连续数帧画面中被检测物体的关联程度以确定是否同一物体，常用方法为卡尔曼滤波和匈牙利算法。

其中，SORT和DeepSORT的核心就是卡尔曼滤波和匈牙利算法。
- 卡尔曼滤波分为两个过程，预测和更新。该算法将目标的运动状态定义为8个正态分布的向量。`预测`：当目标经过移动，通过上一帧的目标框和速度等参数，预测出当前帧的目标框位置和速度等参数。`更新`：预测值和观测值，两个正态分布的状态进行线性加权，得到目前系统预测的状态。
- 匈牙利算法解决的是一个分配问题，在MOT主要步骤中的计算相似度的，得到了前后两帧的相似度矩阵。匈牙利算法就是通过求解这个相似度矩阵，从而解决前后两帧真正匹配的目标。

而SORT算法中只通过前后两帧IOU来构建相似度矩阵，所以运行速度很快。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20200928-1.JPG)

- - -

# Kalman filter


- - -

# Reference

- [https://zhuanlan.zhihu.com/p/59148865](https://zhuanlan.zhihu.com/p/59148865)
- [https://zhuanlan.zhihu.com/p/133678626](https://zhuanlan.zhihu.com/p/133678626)
- [https://zhuanlan.zhihu.com/p/133689982](https://zhuanlan.zhihu.com/p/133689982)
- [Deep SORT??????](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247485748&idx=1&sn=eb0344e1fd47e627e3349e1b0c1b8ada&chksm=9f80b3a2a8f73ab4dd043a6947e66d0f95b2b913cdfcc620cfa5b995958efe1bb1ba23e60100&scene=126&sessionid=1587264986&key=1392818bdbc0aa1829bb274560d74860b77843df4c0179a2cede3a831ed1c279c4603661ecb8b761c481eecb80e5232d46768e615d1e6c664b4b3ff741a8492de87f9fab89805974de8b13329daee020&ascene=1&uin=NTA4OTc5NTky&devicetype=Windows+10+x64&version=62090069&lang=zh_CN&exportkey=AeR8oQO0h9Dr%2FAVfL6g0VGE%3D&pass_ticket=R0d5J%2BVWKbvqy93YqUC%2BtoKE9cFI22uY90G3JYLOU0LtrcYM2WzBJL2OxnAh0vLo)
- [https://zhuanlan.zhihu.com/p/39912633](https://zhuanlan.zhihu.com/p/39912633)
- []()
- []()