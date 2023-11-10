---

layout: post
tags: [Kafka,Distributed System]
title: Apache Kafka分布式消息系统简介
date: 2014-09-26
author: Kaka Chen
comments: true
toc: true
pinned: false

---

# Apache Kafka分布式消息系统简介

## Introduction

Kafka是一个分布式的、分区的、多复本的日志提交服务。它通过一种独一无二的设计提供了一个消息系统的功能。

主要有以下一些消息术语（terminology）需要掌握：
 
 - Kafka用来区分消息的类，成为Topic
 - 向Kafka的主题发布消息的进程成为producer
 - 向Kafka主题注册，并且接收发布到主题的消息的进程成为consumer
 - Kafka是运行在一个集群上的，这个集群由一台或者多台服务器组成，其中每个组成的服务器都被叫做Broker

宏观上讲，producer通过网络向Kafka集群发送消息，而Kafka集群则将这些消息转送服务给consumers，如图：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/1409261.png)

在客户端和服务器之间的交流都是通过一种简单、高效而且与语言无关（language agnostic）的TCP协议来进行。官方文档提供了一个Java携程的Kafka客户端，但是其实这个客户端可以用许多不同语言来写。

所以说Apache Kafka最主要的作用是帮助接收海量数据下的消息，在大数据状态下，收集数据所带来的挑战除了数据量过大外，还需要承担数据分析的功能，通常分为：


 1. 用户行为数据
 2. 应用程序性能跟踪
 3. 日志形式的活动数据
 4. 事件消息

### Topic & Log

Topic是Kafka提供的一种高层抽象，一个topic就是一个类别category或者一个可订阅的条目名称feed naem。对每个topic来说，kafka维护的是如图所示的一个分区日志（partitioned log）：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/1409262.png)

每个分区（partition）是一个有序的、不可变的消息序列，这个序列可以被连续地追加—— 一个提交日志。在分区内的每条消息都有一个有序的id号，这个id号被称为偏移（offset），这个偏移量可以唯一确定每条消息在分区内的位置。
Kafka集群可以在一个指定的时间内保持所有发布上来的消息，不管这些消息有没有被消费。打个比方，如果这个时间设置为两天，那么在消息发布的两天以内，这条消息都是可以被消费的，但是在两天后，这条消息就会被系统丢弃以释放空间。Kafka的性能不会受数据量的大小影响，因此保持大量的数据不是一个问题。

事实上，每个消费者（consumer）唯一需要保存的元数据就是它已经消费到日志里的哪一个位置了，这个位置就是前面提到的偏移量（offset）。这个偏移量是由消费者控制的：一般来说，消费者会按顺序去一条条读日志里的消息，但是如果需要的话，它也是可以重新设置它开始读的位置的。

这些特点的组合意味着消费者（consumer）是很轻量的——它们可以不影响系统性能的情况下来去自如。打个比方，你可以用我们的命令行工具在不影响其他消费者的情况下去抓取任何一个主题的内容。

日志里的分区设计有多个目的。首先，这可以使日志的规模扩大到单独一台机子所能容纳的数量之外。每个单独的分区必须和它所在的主机相匹配，但是一个主题（topic）可以有许多分区，这样的话它就可以处理任意数量的数据。其次这些分区可以作为并行的一个单元——更多的内容稍后会讲到。

### Distributed

日志的分区分布在kafka集群的多台服务器上，每台服务器保存数据以及对每个分区数据的请求。每个分区会在多台服务器上进行副本备份以便容错，副本的数量是可配置的。

每个分区分有一个“leader”的服务器，剩下的都是“follower”。leader处理对这个分区的所有读写请求，与此同时，follower会被动地去复制leader上的数据。如果leader发生故障，其中一个follower会自动成为新的leader。每台服务器可以作为一些分区的leader，同时也作为其他一些分区的follower，这样在集群内部就可以做到一个很好的负载均衡。

### Producer

生产者（producer）可以根据具情况将消息发布（publish）到一个主题（topic）上。生产者可以选择将哪条消息发布到这个主题下的某一个分区（partition）。这可以用传统的轮询方式以保证负载均衡，也可以根据一些语义分区函数来做。更多的有关分区的内容后面会讲到。

### Consumer

传统的消息系统有两种模型：队列和发布-订阅模式。在队列模型中，一堆消费者会从一台机子上读消息，每一条消息只会被一个消费者读到；在发布订阅模型中，消息会向所有的消费者广播。Kafka提供了一种单一的将这两种模型进行抽象的消费者模式——消费者组（consumer group）。

消费者会属于某一个组，发布到每个主题的消息会递送给订阅了这个主题的消费者组中的一个消费者。消费者实例可以是不同的进程或者在不同的机器上。如果所有的消费者从属于同一个组，这就跟传统的队列模式一样了。如果每个消费者自成一组，那么这就是发布订阅模式了，所有的消息会被广播给所有的消费者。但是通常情况下，我们发现，主题会有多个消费者组，每个组对应一个逻辑上的订阅者，每个组由多个消费者实例组成以保证扩展性和容错性。

Kafka处理定序问题上也去传统的消息系统不一样。传统的队列允许消息被任何一个消费者实例所消费。这意味着消息的顺序丢失了。也就是说，尽管在队列中消息是有顺序的，但这些消息会到达不同的消费者并且被无序地处理。在Kafka里，因为有并行的概念——分区，Kafka可以同时提供顺序保证和负载均衡。这可以通过将主题中的分区分配给消费者组中的消费者来实现，这样的话每个分区只被这个组中的一个消费者所消费。这样我们还可以保证这个消费者是这个分区的唯一消费者，并且是按原来的顺序来处理数据的。因为有许多分区，所以也就可以在多个消费者实例中实现负载均衡。需要注意的是，消费者实例的数量不可能比分区多。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/1409263.png)

A two server Kafka cluster hosting four partitions (P0-P3) with two consumer groups. Consumer group A has two consumer instances and group B has four.

### Guaratees

1. 由生产者发送给特定主题分区的消息会以发送的顺序追加（appended）。也就是说，如果一条消息M1被同一个生产者以M2来发送，并且M1先发，那么M1的offset就会比M2更小，并且先出现在日志中。简言之就是先进先出的队列形式
2. 一个消费者实例会按消息在日志中的存储顺序看到它们。
3. 对于副本因子为N的主题，我们可以承受N-1个服务器发生故障而保证提交到日志的消息不会丢失
    
### UseCase

[http://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying](http://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying)




