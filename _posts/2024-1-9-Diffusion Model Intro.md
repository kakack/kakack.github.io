---

layout: post
tags: [Deep Learning, AIGC, Diffusion Model]
title: Diffusion Model Intro
date: 2024-1-9
author: Kaka Chen
comments: true
toc: true
pinned: false

---

# Brief

DDPM的本质作用，就是学习训练数据的分布，产出尽可能符合训练数据分布的真实图片。

当喂给模型一些赛博朋克风格图像，让它学会类似风格图像的信息，再喂给模型一个随机噪音，让根据先前学到的信息再生产一个赛博朋克风格的图像。同理，喂进去的图像可以是人脸，希望模型能从噪声中重建一张人脸图像。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/240109.png)