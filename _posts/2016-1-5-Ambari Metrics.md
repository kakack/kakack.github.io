---

layout: post
categories: [Apache]
tags: [Big Data, Hadoop, Distributed System, Apache Ambari]

---

##Metrics

**Ambari Metrics System（AMS）**最早是在Ambari 2.0.0中引入的，在Ambari所管理的集群中用来收集、聚合和服务Hadoop和系统计量。

- Ambari Metrics System（AMS）：Ambari内置的监控和统计系统
- Metrics Collector：单点运行的收集、聚合、服务计量值的服务器，数据的来源是Metrics Monitor和Metrics Hadoop Service Sinks。
- Metrics Monitor：安装在集群中各个host上，收集计量信息和数据发送给Metrics Collector。
- Metrics Hadoop Sinks：插入多个Hadoop组件池中，然后向Metrics Collector发送Hadoop度量统计信息。

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/16151.jpg)

---

##API

