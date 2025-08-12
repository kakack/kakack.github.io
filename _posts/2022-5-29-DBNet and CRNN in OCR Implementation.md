---
layout: post
tags: [OCR, Deep Learning, Computer Vision]
title: DBNet and CRNN in OCR Implementation
date: 2022-05-29
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

如果将光学字符识别（OCR）技术应用于一些自然场景的图片，从而达到识别图片中文本文字的目的，通常会是一个 two-stage（双阶段） 的过程：

-	文字检测：找到文字在图像中的位置，返回对应位置信息（bounding box，简称 BBox）；
-	文字识别：对 BBox 区域内的文字进行识别，认出其中每个字符，最终将图像中的文字区域转化为可用的字符信息。

### 任务与评测指标

- **文本检测**：常用指标为 Precision、Recall 与 H-mean（F-measure）。检测按 IoU 与文本实例匹配统计 TP/FP/FN。
- **文本识别**：常用指标为字符错误率 CER、词错误率 WER；若为序列到序列任务，可给出准确率（Acc）。
- **端到端 OCR**：综合评估“检测+识别”，通常要求检测框与 GT 对齐同时识别内容匹配。

## DBNet

目前文字检测普遍的做法有两种：

-	基于回归：Textboxes++、R2CNN，FEN等
-	基于分割：Pixel-Link，PSENet，PMTD，LOMO，DBNet等

DBNet 名字中的 DB 指 Differentiable Binarization（可微二值化）模块。传统的分割式文本检测通常先设定固定阈值将概率图二值化，再用启发式聚类获得文本实例；而 DBNet 将“阈值学习”与“二值化”纳入网络端到端联合优化，网络同时输出概率图 P 与阈值图 T，使阈值对不同像素自适应：

\[ \hat{B} = \frac{1}{1 + \exp\left(-k\cdot(P - T)\right)} \] 

其中 \(k\) 为放大系数（常取 50–100），\(\hat{B}\) 为近似二值图。训练时可组合使用 Dice/BCE 对 P 的监督，L1 对 T 的监督，以及对 \(\hat{B}\) 的 BCE 以贴近“收缩标签”（shrink map）。这种设计能在细文本边界与粘连场景下显著提升可分割性。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220528-3.jpeg)

DBNet 整体流程如下：

1.	图像经过 FPN 网络结构，得到四个尺度特征图，分别为 $\tfrac{1}{4}$、$\tfrac{1}{8}$、$\tfrac{1}{16}$、$\tfrac{1}{32}$；
1.	将四个特征图上采样至 $\tfrac{1}{4}$ 并 concat，得到特征图 F；
1.	由 F 预测概率图 P 与阈值图 T；
1.	通过 P、T 计算近似二值图 \(\hat{B}\)。

### 后处理与框生成（Post-processing）

- 二值化与连通域：对 \(\hat{B}\) 进行阈值化与连通域提取；
- 多边形拟合与“外扩”（unclip）：将连通域拟合为多边形/最小外接矩形，并按比例外扩，得到与文本边缘贴合的检测框；
- NMS：对重叠框执行 NMS/框合并，输出最终 BBox/Polygon。

### 训练与实现要点（建议）

- 训练标签：对 GT 多边形做“收缩”（如 pyclipper，收缩比例 0.3–0.5）得到 shrink map；
- 输入分辨率：短边 736/1024，保持长宽比随机缩放与多尺度训练；
- 增广：随机旋转/仿射、颜色抖动、模糊、Mosaic/Copy-Paste（谨慎）；
- 优化与损失：AdamW/SGD，Dice+BCE（P），L1（T），BCE（\(\hat{B}\)）；
- 指标：Precision/Recall/H-mean；同时记录“召回漏检长文本/细文本”的典型失败案例以迭代数据。



![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220528-4.jpeg)



由 F 计算得到 P 和 T 可以表示为：

```Python
1         binary = self.binarize(fuse)   #由F得到P,fuse为特征图F
2         if self.training:
3             result = OrderedDict(binary=binary)
4         else:
5             return binary
6         if self.adaptive and self.training:
7             if self.serial:
9                 fuse = torch.cat(
10                        (fuse, nn.functional.interpolate(
11                            binary, fuse.shape[2:])), 1)
12            thresh = self.thresh(fuse)
```

其中第一行通过 `fuse` 得到 `binary`（即 P），具体实现在 `self.binarize`：

```Python
1         self.binarize = nn.Sequential(
2             nn.Conv2d(inner_channels, inner_channels //
3                       4, 3, padding=1, bias=bias),   #shape:(batch,256,1/4W,1/4H)
4             BatchNorm2d(inner_channels//4),
5             nn.ReLU(inplace=True),  
6             nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2), #shape:(batch,256,1/2W,1/2H)
7             BatchNorm2d(inner_channels//4),
8             nn.ReLU(inplace=True),
9             nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),  #shape:(batch, W, H)
10            nn.Sigmoid())
```





## CRNN



![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220528-1.jpeg)

基于 CNN 和 RNN 的文字识别结构主要有两种形式：

	1.	CNN + RNN + CTC（CRNN + CTC）
	1.	CNN + Seq2Seq + Attention

其中 CRNN 的应用最为广泛。整个网络通常分为三部分：

	-	Conv Layers：卷积网络提取特征映射；
	-	Recurrent Layers：双向 LSTM/GRU 在宽度方向建模序列；
	-	Transcription Layers：CTC 层或全连接输出字符概率分布。

### CTC 解码与训练细节

- 将宽高归一化到固定高度，按列切分得到序列特征；
- 训练采用 CTC Loss，引入 blank（空白）标签以实现对齐自由度；
- 解码常用贪心解码或 Beam Search，可结合语言模型（LM）提升鲁棒性；
- 多语言/多字库时，注意字符集与空格/标点的规范化映射。

### Seq2Seq + Attention 简述

以编码器-解码器结构建模字符序列，解码端通过注意力在特征图上聚焦，逐步输出字符。该方案对弯曲文本、密集排版更友好，但推理延迟与工程复杂度较高。

### 工程化实践与常见问题

- 弯曲/透视文本：在识别前引入 TPS-STN 做几何整形；
- 超长文本：限制最大宽高比并分块滑窗识别；
- 垂直文本与竖排：增加方向分类器或双向解码；
- 检测替代与互补：除 DBNet 外，PSENet/PAN/EAST 等在大文本/密集文本场景表现更佳，可按业务混合使用；
- 评估：除整体 Acc 外，追踪“OCR 失败归因”（检测失败/对齐失败/识别失败），反向驱动数据与模型改进。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220528-2.jpeg)