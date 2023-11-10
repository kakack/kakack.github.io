---

layout: post
tags: [Emacs]
title: Emacs Notes and Tricks
date: 2020-04-14
author: Kaka Chen
comments: true
toc: true
pinned: false

---
# Basic Command

## 文件操作

    - C-x C-f 打开文件
    - C-x C-s 保存文件
    - C-x C-w 存为新文件
    - C-x C-c 退出Emacs

## 编辑操作

    - C-f 前进一个字符
    - C-b 后退一个字符
    - M-f 前进一个字
    - M-b 后退一个字
    - C-a 移到行首
    - C-e 移到行尾
    - M-a 移到句首
    - M-e 移到句尾
    - C-p 后退一行
    - C-n 前进一行
    - M-x goto-line 跳到指定行
    - C-v 向下翻页
    - M-v 向上翻页
    - M-< 缓冲区头部
    - M-> 缓冲区尾部
    - C-l 当前行居中
    - M-n or C-u n 重复操作随后的命令n次
    - C-d 删除一个字符
    - M-d 删除一个字
    - C-k 删除一行
    - M-k 删除一句
    - C-w 删除标记区域
    - C-y 粘贴删除的内容
    - 注意：C-y可以粘贴连续C-k删除的内容；先按C-y，然后按M-y可以选择粘贴被删除的内容
    - C-空格 标记开始区域(需修改输入法快捷键)
    - C-x h 标记所有文字
    - M-w 复制标记区域
    - C-/ or C-x u 撤消操作

## 执行SHELL命令

    - M-x shell 打开SHELL
    - M-! 执行SHELL命令 (shell-command)

## 窗口操作

    - C-x 0 关闭本窗口
    - C-x 1 只留下一个窗口
    - C-x 2 垂直均分窗口
    - C-x 3 水平均分窗口
    - C-x o 切换到别的窗口
    - C-x s 保存所有窗口的缓冲
    - C-x b 选择当前窗口的缓冲区
    - C-M v 另外一个窗口向下翻页(需要对照时很好用)
    - c-M-Shift v 另外一个窗口向上翻页

## 缓冲区列表操作

    - C-x C-b 打开缓冲区列表
    - C-x k 关闭缓冲区

## 搜索模式

    - C-s 向前搜索
    - C-s 查找下一个
    - ENTER 停止搜索
    - C-r 反向搜索
    - C-s C-w 以光标所在位置的字为关键字搜索
    - M-x replace-string ENTER search-string ENTER 替换
    - C-M-s 向前正则搜索
    - C-M-r 向后正则搜索
    - C-M-% 正则交互替换

## 帮助
    
    - C-h t 入门指南
    - C-h v 查看变量
    - C-h f 查看函数
    - C-h ? 查看帮助列表

- - -

# Use SML Mode

1. 进入SML mode: M-x sml-mode ；
2. 创建一个*sml的缓冲：C-x C-f *.sml + Reruen/Enter(这儿的*是指文件名)；
3. 如果已经有sml文件了，直接拖进Emacs里；
4. 在光标处写入 val x = 3 + 4 （换行）val y = x * 5；
5. 用C-x C-s 保存；
6. 运行：可以在这个缓冲区运行，C-c C-s + Return，可以看到缓冲器被分离开了。这时，可以在光标处写入一些式子，例如1 +1 ；;Return，可以看到结果；
7. 在光标处输入use "my.sml"; + Return（my就是当前的文件名），皆可以看到文件里面的程序运行的结果；
8. 如果要退出 sml mode，输入C-d。