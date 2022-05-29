---
layout: post
categories: [Computer Vision]
tags: [OCR, Deep Learning]


---



# Intro

如果将光学字符识别（OCR）技术应用于一些自然场景的图片，从而达到识别图片中文本文字的目的，通常会是一个two-stage的过程：

	-	文字检测：找到文字在图像中的位置，返回对应位置信息（Bounding Box）；
	-	文字识别：对于对应Bounding Box区域内的文字进行识别，认出其中每个文字字符代表什么，最终将图像中的文字区域转化为字符信息。



# DBNet

目前文字检测普遍的做法有两种：

	-	基于回归：Textboxes++、R2CNN，FEN等
	-	基于分割：Pixel-Link，PSENet，PMTD，LOMO，DBNet等

DBNet名字中的DB指Differentiable Binarization，即微分二值化的模块，

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220528-3.jpeg)

# CRNN



![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220528-1.jpeg)

基于CNN和RNN的文字识别结构主要有两种形式：

	1.	CNN+RNN+CTC(CRNN+CTC)
	1.	CNN+Seq2Seq+Attention

其中CRNN的应用最为广泛。整个CRNN网络可以分为三个部分：

	-	Conv Layers：即普通CNN网络用于提取图像Conv Feature Maps；
	-	Recurrent Layers：一个深层双向LSTM网络，在卷积特征的基础上继续提取文字序列特征；
	-	Transcription Layers：将RNN输出做softmax后，为字符输出。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20220528-2.jpeg)