---

layout: post
categories: [Spark]
tags: [Big Data, Distributed System, Spark Streaming]

---

在看例子展示描述的过程中，发现两个对Spark Streaming和Apache Storm做详细对比的ppt：

- [雅虎的Bobby Evans 和Tom Graves的个"Spark and Storm at Yahoo!"演讲](http://www.slideshare.net/ptgoetz/apache-storm-vs-spark-streaming)

- [Hortonworks的P. Taylor Goetz也分享过名为 Apache Storm and Spark Streaming Compared的讲义](http://www.slideshare.net/ptgoetz/apache-storm-vs-spark-streaming)

- - -

#[Link 1](http://www.slideshare.net/ptgoetz/apache-storm-vs-spark-streaming)

关于二者内容的介绍在之前博文中有，在此不再赘述。详细讲一些比较内容。本着“选合适的工具做合适的工作”的原则，需要考量的要素有：

- **Scale范围**
- **Latency延迟**
- **Iterative Processing迭代过程**
- **Use What You Know用你会的**
- **Code Reuse代码重用性**
- **Maturity成熟性**

- - -

###什么时候用Spark Streaming？

- 迭代Batch过程（大多数Machine Learning）
- 可靠的ETL（Extract-Transform-Load ）
- 可靠的Shark/交互查询

- - -

- <1TB（或者本身cluster的内存大小）
- 调整到正常运行比较痛苦
- 数据块或者其他在大规模上工作

- - -

- Streaming都是μ-batch，所以延时最多1s
- Streaming有单点失败的存档
- 所有Streaming输入都在内存中有备份

- - -
###什么时候用Apache Storm

- 延迟<1s，每次都是单个事件
- 即时性很高（分析、预算、ML、anything）

- - -

- 但是API比Spark的来的低级
- 缺乏内置的回看集成概念
- 需要花费更多精力去把batch和Streaming结合起来

- - -

#[Link 2](http://www.slideshare.net/ptgoetz/apache-storm-vs-spark-streaming)

观点1：在本职上，Streaming和Batch处理就是不同的。

- **Storm**是一个stream processing框架，也会做一些micro-batch（如Trident）
- **Spark**是一个batch processing框架，也会做一些micro-batch（如Spark Streamming）

所以对于stream和batch来说是两个互相独立又有部分重合的概念，重合部分就是所谓的micro-batch。

- - -
##Apache Storm：有两个Streaming APIs

- Core Storm（Spouts and Bolt）
	- Once at a time
	- 低延时
	- 在Tuple Streams上操作

- Trident（Streams and Operations）
	- Micro-Batch
	- 更高的吞吐量
	- 在由Tuple Batch和Partitions组成的Streams上进行操作  

- - -

##支持的语言类型

Core Storm | Storm Trident |Spark Streaming
-----------|---------------|---------------
Java|Java|Java
Clojure|Clojure|Scala
Scala|Scala|Python
Python|
Ruby|
Others*|

- - -

##可靠性模型


Item|Core Storm|Storm Trident|Spark Streaming
----|----------|-------------|---------------
At Most Once|Yes|Yes|No
At Least Once|Yes|Yes|No
Exactly Once|No|Yes|Yes
 
- - -

##编程模型


Item|Core Storm|Storm Trident|Spark Streaming
----|----------|-------------|---------------
Stream Primitive|Tuple|Tuple,Tuple Batch,Partition|DStream
Stream Source|Spout|Spouts,Trident Spouts|HDFS, Network Socket and so on
Computation/Transformation|Bolts|Filters, Functions, Aggregations, Joins|Transformation, Window Operations
Stateful Operations|No(roll your own)|Yes|Yes
Output/Persistence|Bolts|State, Map State|foreachRDD


- - -

##Support


Item|Core Storm|Storm Trident|Spark Streaming
----|----------|-------------|---------------
Hadoop Distro|Hortonworks, MapR|Cloudera, MapR, Hortonworks(preview)|Cloudera, MapR, Hortonworks
Resource Management|Yarn, Mesos|Yarn, Mesos|Yarn, Mesos
Provisioning Monitoring|Apache Ambari|Cloudera Manager|?


- - -

##Failure场景

关于Spark Streaming的worker failure：如果一个worker node坏了，那么系统会从之前剩下的input备份数据中重新计算结果，如果一个作为network receiver的worker node坏了，那么意味着有一部分数据丢失了，那就意味着这些数据被认为已经被系统接收但是没有被备份到其他节点上。总而言之，只有HDFS-backed数据源是被认为完全fault tolerant的。一般认为，解决方案是写ahead logs。

关于Apache Storm：如果一个supervisor node坏了，那么Nimbus将会重新分配节点的任务到cluster上其他节点；任何tuple被发送到一个坏了的节点会造成超时情况，那么这个tuple会被重新处理（在Trident中，任何batch都会被重新处理）；传输保证取决于一个可靠的数据源。

- - -

##数据源的可靠性

- 当一个数据源被认为**unreliable**，那就是其中没有任何方法能够将一个之前接收到的消息做重新处理。
- 一个数据源被认为**reliable**，那就是在处理过程中发生任何的错误都可以有一定的办法进行回滚和重新处理。
- 一个数据源被认为是**durable**，那就意味着在任何情况和标准下，都可以对消息或者消息组进行回滚和重处理。

所以对Apache Storm的可靠性限制：

- Exactly once processing要求一个durable的数据源；
- At least once processing需要一个reliable的数据源；
- 一个unreliable的属于原可以被包装起来去提供额外的保障；
- 通过durable和reliable的数据源，Storm不会丢失数据；
- 通用模式：通过Apache Kafka来返回unreliable数据源，损失一点延时来获取100%的durability。

对于Spark Streaming的可靠性限制：

- 所有容错和可靠性保障都要求一个HDFS-backed数据源；
- 将数据在Stream处理之前，先存放到HDFS上，这么做可能会带来一点延时；
- 网络数据源（比如Kafka）在一个worker node failure上都是十分脆弱容易丢失的
 
 
 
 
 
 
 
 
 
 