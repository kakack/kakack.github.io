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

和传统虚拟机的比较
----

容器和虚拟机都有类似的资源独立和分配的优势，但是在架构上也有些许区别，相比较而言，容器更加易扩展和轻便有效。

![传统虚拟机](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/16-6-21-WhatIsDocker_2_VMs.png)

传统虚拟机包括了需要运行的应用程序，和其所必要的二进制文件和类库，以及一整个guest操作系统，所有的这些加起来可能会有十几GB。


![容器](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/16-6-21-WhatIsDocker_3_Containers.png)

容器包含了这个应用程序和它所有的依赖，但是与其他容器共享一个内核kernel，在host操作系统上以一个独立进程的形式在用户空间上运行。Docker容器并不和任何特定的基础设施绑定，它们可以在任何电脑或者任何基础设施上运行，或者任何云。

与传统VM相比，直观上容器最大的不同是被设计成用来运行单进程，无法模拟一个完整的环境。

所以总结来说，相对于普通的VM，容器有以下优势：

- 启动速度快，容器通常在一秒内可以启动，而 VM 通常要更久
- 资源利用率高，一台普通 PC 可以跑上千个容器，你跑上千个 VM 试试
- 性能开销小， VM 通常需要额外的 CPU 和内存来完成 OS 的功能，这一部分占据了额外的资源

在容器中，跳过了设置和管理开发环境、特定语言环境等，专注于创建新的特征功能、修正问题和分发软件。

- 隔离应用依赖
- 创建应用镜像并进行复制
- 创建容易分发的即启即用的应用
- 允许实例简单、快速地扩展
- 测试应用并随后销毁它们


核心组件
====

Docker Engine
---
Docker Engine是一个client-sever结构的应用，具有这些组件：

- 一个sever，是一个一直在运行的程序，叫做daemon process
- 一个REST API用来指定特定的接口，程序可以使用这些接口来和daemon交流并指挥它做一些工作
- 一个命令行工具client

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/16-6-22engine-components-flow.png)



核心架构
---
Docker使用一个client-server架构，Docker client和Docker daemon相互沟通，二者可以在同一个系统上运行，也可以用client连接到一个远程的Docker daemon。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/16-6-22-architecture.png)

Docker Daemon
---
Daemon早host machine上运行，用户不必直接和daemon交互，而是由client与之交互。Daemon上进行一些building、运行、分发容器等工作。

Docker Client
---
用户操作接口，接受命令，和后端Daemon交互。

Docker内部
---
- Docker image：一个只读的模板，用来创建Docker容器。是Docker的**build**组件。
- Docker registries：拥有Docker images，是public或者private的image储存。是Docker的**distribution**组件。
- Docker containers：类似于一个目录，包含了一个应用要运行的所有需要的内容，每个容器都是由一个image创建出来的。一个容器可以被运行、开始、停止、移动和删除。每一个容器都是一个独立而安全的应用平台。是Docker的**run**组件。


在Macbook上的安装和使用
===





