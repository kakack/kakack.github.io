---

layout: post
categories: [docker]
tags: [Linux, docker]

---

什么是Docker
===

Docker简单说是一种世界领先的软件集装箱化运行的平台。是一个开源的引擎，可以轻松的为任何应用创建一个轻量级的、可移植的、自给自足的容器。开发者在笔记本上编译测试通过的容器可以批量地在生产环境中部署，包括VMs（虚拟机）、bare metal、OpenStack 集群和其他的基础应用平台。 

Docker的应用场景主要有以下几个：

- web应用的自动化打包和发布；
- 自动化测试和持续集成、发布；
- 在服务型环境中部署和调整数据库或其他的后台应用；
- 从头编译或者扩展现有的OpenShift或Cloud Foundry平台来搭建自己的PaaS环境。

Docker内把一系列的软件打包到一个完整的文件系统内，包含了所需要运行的所有内容：代码、运行时间、系统工具、系统类库等，这保证了整个软件不管在什么环境下都能一样运行。

##和传统虚拟机的比较

容器和虚拟机都有类似的资源独立和分配的优势，但是在架构上也有些许区别，相比较而言，容器更加易扩展和轻便有效。

![传统虚拟机](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/16-6-21-WhatIsDocker_2_VMs.png)

传统虚拟机包括了需要运行的应用程序，和其所必要的二进制文件和类库，以及一整个guest操作系统，所有的这些加起来可能会有十几GB。


![容器](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/16-6-21-WhatIsDocker_3_Containers.png)

容器包含了这个应用程序和它所有的依赖，但是与其他容器共享一个内核kernel，在host操作系统上以一个独立进程的形式在用户空间上运行。Docker容器并不和任何特定的基础设施绑定，它们可以在任何电脑或者任何基础设施上运行，或者任何云。