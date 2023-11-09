---

layout: post
categories: [Python]
tags: [Jupyter Notebook, Python, Mac, Linux]
title: Jupyter Notebook在Mac和Linux上的配置及远程访问
date: 2017-12-05
author: Kaka Chen
comments: true
toc: true
pinned: true

---

# IPython 和 Jupyter

`IPython` 通常指的是一个 `Python REPL(交互式解释器) shell`。提供了远比 `Python shell` 强大的 `shell` 环境。`IPython` 是 `Iteractive Python shell `的缩写。 `Notebook` 是一个基于 `IPython` 的 `web` 应用。

截止 `IPython 3.0` ，`IPython` 变得越来越臃肿，因此， `IPython 4.x` 后，`IPython`被分离成为了`IPython kernel's Jupyter（IPython shell combined with jupyter kernel）` 和`Jupyter subprojects` ，`Notebook` 是 `Jupyter` 的一个 subproject 。官方将此称为`The Big Split™。#reference-The Big Split™`

`IPython` 把与语言无关`（language-agnostic）`的 `components` 和 `Python` 解释器剥离，独立为`Jupyter`发行。因此，`Jupyter` 可以和 `Python` 之外的解析器结合，提供新的、强大的服务。比如 `Ruby REPL` 环境 `IRuby` 和 `Julia REPL` 环境 `IJulia`。

因此，`Jupyter` 和 `IPyhon` 指代同一个项目， `Jupyter` 特指 `Ipython4` 后的， 非`python kernel`框架。

`Jupyter Notebook`已经逐渐取代IDE成为了多平台上写简单`Python`脚本或应用的几家选择。

`Jupyter Notebook`可以通过`pip/pip3`安装：

```shell
sudo pip3 install jupyter
```

然后在目标文件夹目录下，输入指令`jupyter notebook`开启服务，可在浏览器地址`localhost:8888`中访问主页

- - -

# 允许远程访问

我之前在训练一些简单的svm模型的时候，因为数据量大，训练时间慢，导致自己的macbook在训练过程中一直无法做别的事情，就动了在远程Linux虚拟机上装一套`Jupyter Notebook`然后将耗时很长的训练操作放在远程机器上做。

在本地，我们访问`localhost:8888`就能看到`Jupyter Notebook`的本地主页，但是在远程访问中，并不能直接这么做。因此需要以下一些操作：

[官方指南](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html#notebook-server-security)

### 1. 生成一个 notebook 配置文件

默认情况下，配置文件 `~/.jupyter/jupyter_notebook_config.py` 并不存在，需要自行创建。使用下列命令生成配置文件：

```shell
jupyter notebook --generate-config
```

如果是 root 用户执行上面的命令，会发生一个问题：

```
Running as root it not recommended. 
Use --allow-root to bypass.
```

提示信息很明显，root 用户执行时需要加上 --allow-root 选项。

```shell
jupyter notebook --generate-config --allow-config
```

执行成功后，会出现下面的信息：

```
Writing default config to: /root/.jupyter/jupyter_notebook_config.py
```

### 2. 生成密码

#### 自动生成

从 jupyter notebook 5.0 版本开始，提供了一个命令来设置密码：`jupyter notebook password`，生成的密码存储在 `jupyter_notebook_config.json`。

```
$ jupyter notebook password
Enter password:  ****
Verify password: ****
[NotebookPasswordApp] Wrote hashed password to /Users/you/.jupyter/jupyter_notebook_config.json
```

#### 手动生成

除了使用提供的命令，也可以通过手动安装，我是使用的手动安装，因为`jupyter notebook password` 出来一堆内容，没耐心看。打开 ipython 执行下面内容：

```
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password:
Verify password:
Out[2]: 'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
```

`sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed` 这一串就是要在 `jupyter_notebook_config.py` 添加的密码。

```
c.NotebookApp.password = \
u'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
```

### 3. 修改配置文件

在 `jupyter_notebook_config.py` 中找到下面的行，取消注释并修改。

```
c.NotebookApp.ip='*'
c.NotebookApp.password = u'sha:ce... # 刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 # 可自行指定一个端口, 访问时使用该端口
```

以上设置完以后就可以在服务器上启动 `jupyter notebook，jupyter notebook`, root 用户使用 `jupyter notebook --allow-root`。打开 IP:指定的端口(默认为8888), 输入密码就可以访问了。

- - -

# 同时支持python2和python3

先安裝Python2和Python3的ipython notebook

```shell
pip2 install ipython notebook
pip3 install ipython notebook
```

分别用各自的ipython执行下面的指令

```shell
ipython2 kernelspec install-self
ipython3 kernelspec install-self
```

就能在ipython notebook里面同时使用两种版本的Python了

附录：[27 个Jupyter Notebook的小提示与技巧](http://liuchengxu.org/pelican-blog/jupyter-notebook-tips.html)
