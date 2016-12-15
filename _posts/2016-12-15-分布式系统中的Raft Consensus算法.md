---

layout: post
categories: [Hadoop]
tags: [Hadoop, Distributed System,  Algorithm]

---

#分布式系统中的Raft Consensus算法

以Kudu为例子，在复制和更替Partition的时候，常用的算法就是Raft Consensus。其中Consensus是指在多个服务器正常工作状态下，出现一个或若干个节点宕机，影响整个系统功能和稳定，因此需要有一些替换策略将宕机了的节点功能转移到其他节点，来保证整个系统的容错性和稳定性。整个替换策略中，只要有半数的节点达成一致就能完成替换。

整个过程很类似于现实社会中的选举，参选的节点需要说服选民节点们把票投给自己，一旦当选就可以完成接下去的操作。

在整个过程中，任何一个节点都会成为以下三个roles中之一：

- **Leader**：处理客户端交互，catalog复制等，一般整个过程中只有一个Leader
- **Follower**：类似选民，完全被动，有选举权
- **Candidate**：当需要一个新Leader时，可以成为被选举的对象


整个Raft过程分为选举和选举后操作两个过程。

---

### 选举过程

任何节点都可以成为一个candidate，当需要一个新leader时，可以要求其他节点选自己。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft1.jpg)

其他节点同意，返回ok消息。



![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft2.jpg)

这个过程中如果有follower宕机，candidate仍可以自己投票给自己。

一旦candidate当选leader，可以向follower们发出指令

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft3.jpg)

然后通过心跳进行日志复制的通知

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft4.jpg)

如果当这个leader宕机了，那么Follower中会出现一个candidate，发出选举请求。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft5.jpg)

Follower同意之后，成为Leader，继续承担日志复制工作

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft6.jpg)

整个选举过程中，需要一个时间限制，如果两个candidate同时发出投票邀请，可以通过加时赛来解决。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft7.jpg)

---

### 日志复制

假设Leader已经被选出，这时客户端邀请增加一个日志，叫“Sally”。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft8.jpg)

Leader要求各个Follower把日志追加到各自日志中。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft9.jpg)

大多数Follower完成操作，追加成功，返回commited OK。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/raft/raft10.jpg)

在下一个心跳heartbeat中，Leader会通知所有Follwer更新commited 项目。




