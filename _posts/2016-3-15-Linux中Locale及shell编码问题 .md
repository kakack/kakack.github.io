---

layout: post
categories: [Linux]
tags: [Linux]

---

参考[Linux中Locale及shell编码问题](http://www.drupal001.com/2012/04/shell-utf8/)

- - -

这段时间在Ubuntu上用Tab补全路径或文件名的时候一直出现以下错误：

```
-bash: warning: setlocale: LC_CTYPE: cannot change locale (UTF-8)
```

Locale是linux系统中多语言环境的设置接口，Locale根据计算机用户所使用的语言，所在国家或者地区，以及当地的文化传统所定义的一个软件运行时的语言环境。
locale把按照所涉及到的文化传统的各个方面分成12个大类，这12个大类分别是：

1. 语言符号及其分类(LC_CTYPE)
2. 数字(LC_NUMERIC)
3. 比较和排序习惯(LC_COLLATE)
4. 时间显示格式(LC_TIME)
5. 货币单位(LC_MONETARY)
6. 信息主要是提示信息,错误信息,状态信息,标题,标签,按钮和菜单等(LC_MESSAGES)
7. 姓名书写方式(LC_NAME)
8. 地址书写方式(LC_ADDRESS)
9. 电话号码书写方式(LC_TELEPHONE)
10. 度量衡表达方式 (LC_MEASUREMENT)
11. 默认纸张尺寸大小(LC_PAPER)
12. 对locale自身包含信息的概述(LC_IDENTIFICATION)。

命令`locale -a`能查看这些参数

命令`locale-gen`生成并更新，或者用`locale-gen en_US.UTF-8`指定某个更新

在我的Ubuntu中，保存这部分内容的文件在`\etc\locale.alias`中，可以添加上以下两行

```
en_US.UTF-8 UTF-8  
en_US ISO-8859-1
```

再运行一遍`locale-gen`完成更新

- - -

另在原作中提到一个办法：

```
echo $SHELL
#获得SHELL的path，在此我的是/bin/bash
sudo chsh -s /bin/bash root
#后两个参数是shell的path和用户名
source .bashrc
```

- - -

还提及的一个办法是在`~/.profile`中添加`export LC_CTYPE="en_US.UTF-8"`

也有网友说，应该添加以下三行：

```
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

最后执行`source ~/.profile`

具体效果可以根据实际来做。





