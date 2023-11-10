---

layout: post
tags: [ARM]
title: ARM Component GIC
date: 2023-11-06
author: Kaka Chen
comments: true
toc: true
pinned: false

---

GIC: Generic interrupt controller，是用于在外设和core之间，以及core与core之间处理中断的控制器

GIC Implementation	| GIC Architecture Specification
---|---
GIC-400 [CoreLink GIC-400 Generic Interrupt Controller Technical Reference Manual r0p1 (arm.com)](https://developer.arm.com/documentation/ddi0471/b/introduction/about-the-gic-400?lang=en)	|[GICv2](https://developer.arm.com/documentation/ihi0048/bb/?lang=en)
GIC-500 [ARM CoreLink GIC-500 Generic Interrupt Controller Technical Reference Manual r1p1](https://developer.arm.com/documentation/ihi0069/d/?lang=en)	|GICv3 [ARM Generic Interrupt Controller Architecture Specification GIC Architecture Version 3.0 and 4.0](https://developer.arm.com/documentation/ihi0069/d/?lang=en)
GIC-600 [Arm CoreLink GIC-600 Generic Interrupt Controller Technical Reference Manual r1p6](https://developer.arm.com/documentation/100336/0106/introduction/about-the-gic-600?lang=en)	|GICv3
GIC-600AE [Arm CoreLink GIC-600AE Generic Interrupt Controller Technical Reference Manual](https://developer.arm.com/documentation/101206/0003/?lang=en)	|GICv3
GIC-625 [Arm CoreLink GIC-625 Generic Interrupt Controller Technical Reference Manual](https://developer.arm.com/documentation/102143/0001/?lang=en)	|GICv3, GICv3.1
GIC-700 [Arm CoreLink GIC-700 Generic Interrupt Controller Technical Reference Manual](https://developer.arm.com/documentation/101516/0202/About-the-GIC-700?lang=en) |GICv3, GICv3.1, GICv4.1 [Arm Generic Interrupt Controller Architecture Specification GIC Architecture Version 3.1 and 4.1](https://developer.arm.com/documentation/ihi0069/g)


# Brief

GIC v3架构被设计到用于运行ArmV8-A和ArmV8-R架构编译的处理组件（Processing Elements， PE），架构定义了：

- 处理连接到GIC上的任何PE的所有中断源的架构要求；
- 适用于单处理器或多处理器系统的通用中断控制器编程接口

GIC提供：

- 跟中断源、中断行为，以及中断路由有关的寄存器
- 支持：
	- The Armv8 architecture.
	- Locality-specific Peripheral Interrupts (LPIs).
	- Private Peripheral Interrupts (PPIs).
	- Software Generated Interrupts (SGIs).
	- Shared Peripheral Interrupts (SPIs).
	- Interrupt masking and prioritization.
	- Uniprocessor and multiprocessor systems.
	- Wakeup events in power management environments.

ARM-cortex A系列处理器中，提供了4个管脚给soc（System on Chip），实现外界中断的传递（其中后两个虚拟中断暂时不聊）
- nIRQ： 物理普通中断
- nFIQ: 物理快速中断
- nVIRQ: 虚拟普通中断
- nVFIQ: 虚拟快速中断
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231010103628.png)

当中断有效时，GIC根据中断信号中的配置决定是否发送给CPU。如果有多个有效中断信号，GIC还会根据优先级进行仲裁。CPU收到GIC发送的中断，可以通过访问GIC的寄存器得知中断来源和后续操作。当处理完毕后CPU会通知GIC，GIC接到通知后取消该中断源。

---
# About Interruption

关于中断的**状态**：

- inactive：中断处于无效状态
- pending：中断处于有效状态，但是cpu没有响应该中断
- active：cpu在响应该中断
- active and pending：cpu在响应该中断，但是该中断源又发送中断过来
![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231010231803.png)

中断的**触发方式**：
- `edge-triggered`：边沿触发，当中断源产生一个边沿，中断有效
- `level-sensitive`：电平触发，当中断源为指定电平，中断有效

中断**认可**：CPU响应该中断，中断状态由pending变为active，可以通过访问GICC_IAR寄存器进行中断认可：
- `GICC_IAR`： 认可group0的中断
- `GICC_AIAR`：认可group1的中断

中断**类型**：
- `PPI（private peripheral interrupt）`，私有外设中断，该中断来源于外设，但是该中断只对指定的core有效。
- `SPI（shared peripheral interrupt）`，共享外设中断，该中断来源于外设，但是该中断可以对所有的core有效。
- `SGI（software-generated interrupt）`，软中断，软件产生的中断，用于给其他的core发送中断信号
- `virtual interrupt`，虚拟中断，用于支持虚拟机

中断**优先级**：在 gic 中，优先级由 8 位表示，总共 256 个等级，但实际应用中用不了那么多，常用4位表示16个等级，其中value值越小表示优先级越高。

为了方便中断管理，gic为每个中断分配了`中断号（interrupt ID）`：
- `ID0-ID15`，分配给SGI
- `ID16-ID31`，分配给PPI
- `ID32-ID1019`，分配给SPI
- `Other`

中断**完成**：状态从activate变成inactivate，GIC对此定义了两个stage：
- 优先级重置（priority drop）：将当前中断屏蔽的最高优先级进行重置，以便能够响应低优先级中断。group0中断，通过写GICC_EOIR寄存器，来实现优先级重置，group1中断，通过写 GICC_AEOIR 寄存器，来实现优先级重置。
- 中断无效（interrupt deactivation）：将中断的状态，设置为inactive状态。通过写 GICC_DIR 寄存器，来实现中断无效。

GIC中还有一个功能称为`banking`，包括两个功能：
- `中断banking`，对于PPI和SGI，gic可以有多个中断对应于同一个中断号。比如在soc中，有多个外设的中断，共享同一个中断号。
- `寄存器banking`，对于同一个gic寄存器地址，在不同的情况下，访问的是不同的寄存器。例如在secure和non-secure状态下，访问同一个gic寄存器，其实是访问的不同的gic的寄存器。

---
# GIC核心组件

## Distributor
中断分发器，用来收集所有的中断来源，并且为每个中断源设置中断优先级，中断分组，中断目的core。当有中断产生时，将当前最高优先级中断，发送给对应的cpu interface。Distributor对中断提供以下的功能：
- 全局中断使能
- 每个中断的使能
- 中断的优先级
- 中断的分组
- 中断的目的core
- 中断触发方式
- 对于SGI中断，传输中断到指定的core
- 每个中断的状态管理
- 提供软件，可以修改中断的pending状态

## CPU Interface
cpu interface，将GICD发送的中断信息，通过IRQ，FIQ管脚，发送给连接到该cpu接口的core，提供了一下的功能：
- 将中断请求发送给cpu
- 对中断进行认可（acknowledging an interrupt）
- 中断完成识别(indicating completion of an interrupt)
- 设置中断优先级屏蔽
- 定义中断抢占策略
- 决定当前处于pending状态最高优先级中断

## Virtual CPU Interface
将GICD发送的虚拟中断信息，通过VIRQ，VFIQ管脚，传输给core，每个core有一个virtural cpu Interface，其中又包括两个组件：
- `virtual interface control`：寄存器使用 GICH_ 作为前缀；
- `virtual cpu interface`：寄存器使用 GICV_ 作为前缀。

---
# GIC v2

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231010234843.png)

## GICv2例子
以下为一个gicv2例子，GICv2结构只支持8core。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106153456.png)

外部中断连接到gic由gic distributor进行中断分组，分组后的中断请求由distributor发送给gic内的CPU Interface，再发送给processor。对于支持安全扩展，其应用如下（安全中断，处于group0，非安全中断处于group1。）：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106154027.png)

## GICv2寄存器

GICv2的寄存器均是以`memory-map`形式访问：

### 1. distributor register

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106154542.png)

### 2. cpu interface register

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106154608.png)

---
# GIC v3

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231010234918.png)

- `Distributor`：SPI中断的管理，将中断发送给redistributor
- `Redistributor`：PPI，SGI，LPI中断的管理，将中断发送给cpu interface
- `CPU interface`：传输中断给core
- `Interrupt Translation Service(ITS)`：用来解析LPI中断

其中distributor、redistributor和ITS是实现在GIC内部的，cpu interface是在core内部的。
## 在GICv2上的更新

- 使用属性层次（affinity hierarchies），来对core进行标识，使gic支持更多的core
- 将cpu interface独立出来，用户可以将其设计在core内部
- 增加redistributor组件，用来连接distributor和cpu interface
- 增加了LPI，使用ITS来解析
- 对于cpu interface的寄存器，增加系统寄存器访问方式


## 属性层次

GICv3对core不再使用单一数字表示，而是用属性层次来标识，如`xxx.xxx.xxx.xxx`，类似于arm core用`MPIDR_EL1`系统寄存器来标识core。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106175113.png)

每个core连接一个cpu interface，而cpu interface会连接gic中的一个redistributor，redistributor的标识和core的标识一样。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106180500.png)


## GICv3 中断处理流程

中断流程有两类：
- 通过distributor，如SPI；
	- 外设发起中断给distributor；
	- distributor将该中断分发给合适的redistributor；
	- cpu interface产生合适的中断异常给processor；
	- 处理器接受该异常并且由软件处理该中断
- 不通过distributor，如LPI
	- 外设发起中断给ITS；
	- ITS分析中断决定该分配的redistributor；
	- ITS将中断分配给合适的redistributor；
	- redistributor将中断信息发给cpu interface；
	- cpu interface产生合适的中断异常给processor；
	- processor接收该异常，并且由软件处理该中断。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106181418.png)

## GICv3寄存器

GICv3对寄存器有两种访问方式：
- Memory-mapped（实现在GIC内部）
	- - GICC: cpu interface寄存器
	- GICD: distributor寄存器
	- GICH: virtual interface控制寄存器，在hypervisor模式访问
	- GICR: redistributor寄存器
	- GICV: virtual cpu interface寄存器
	- GITS: ITS寄存器
- 系统寄存器访问（实现在core内部）
	- ICC: 物理 cpu interface 系统寄存器  
	- ICV: 虚拟 cpu interface 系统寄存器
	- ICH： 虚拟 cpu interface 控制系统寄存器

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106182828.png)

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20231106183047.png)

---
# Reference
 - [arm GIC介绍之一](https://blog.csdn.net/sunsissy/article/details/73791470)
 - [深度剖析ARM中断控制器与GIC中断控制器](https://zhuanlan.zhihu.com/p/527107797)
 - [linux中断子系统-arm-gic 介绍](https://zhuanlan.zhihu.com/p/363129733)
 - [ARM GIC（一） cortex-A 处理器中断简介](https://www.cnblogs.com/schips/p/arm-gic-serial.html)

