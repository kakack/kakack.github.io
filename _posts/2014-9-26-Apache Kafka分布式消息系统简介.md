---

layout: post
tags: [Kafka, Distributed System]
title: Apache Kafka分布式消息系统简介
date: 2014-09-26
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

Kafka是一个分布式的、分区的、多副本的分布式日志（commit log）服务，同时以独特的架构提供消息系统能力。

主要有以下一些消息术语（terminology）需要掌握：
 
 - 用来区分消息类别的抽象，称为 Topic；
 - 向 Kafka 的主题发布消息的进程称为 producer；
 - 订阅主题并接收消息的进程称为 consumer；
 - Kafka 运行在由一台或多台服务器组成的集群上，集群中的每台服务器称为 broker。

宏观上讲，producer 通过网络向 Kafka 集群发送消息，而 Kafka 集群将这些消息转送给 consumers，如图：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/1409261.png)

在客户端与服务器之间的通信采用一种简单、高效且与语言无关（language agnostic）的基于 TCP 的二进制协议。官方提供 Java 版本的 Kafka 客户端，同时也有多语言生态实现。

所以说 Apache Kafka 最主要的作用是支撑海量数据场景下的消息流转与数据采集。在大数据语境中，数据采集的挑战除了流量巨大，还包括实时处理与下游分析的解耦，通常包括：

 
  1. 用户行为数据；
  2. 应用程序性能跟踪；
  3. 日志形式的活动数据；
  4. 事件消息。

### Topic & Log

Topic 是 Kafka 提供的一种高层抽象，一个 topic 可以理解为一个类别（category）或者一个可订阅的条目名称（feed name）。对每个 topic 来说，Kafka 维护的是如图所示的一个分区日志（partitioned log）：

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/1409262.png)

每个分区（partition）是一个有序且不可变的消息序列，序列支持顺序追加（append-only）—— 即提交日志。在分区内的每条消息都有一个单调递增的 id（offset），唯一确定消息在分区内的位置。
Kafka 集群可在指定的保留策略内（按时间或按总大小）保留消息，不论是否被消费。例如保留两天，则两天内的消息均可被消费，两天后消息会被删除以释放空间。Kafka 的顺序读写与页缓存使其性能不随历史数据量线性下降，因此保留海量历史数据不是问题。

事实上，每个消费者（consumer）唯一需要保存的元数据就是它已消费到日志里的位置（offset）。这个 offset 由消费者控制：通常按顺序读取，但也可以根据需要重置读取位置（如从最早 earliest 或最新 latest 开始）。

这些特点的组合意味着消费者是轻量的——可以随时加入或离开而不影响其他消费者。例如可用命令行工具在不影响其他消费者的情况下抓取任意主题内容。

日志里的分区设计有多个目的。首先，它允许主题的存储规模横向扩展到单机之外，每个分区物理上属于某个 broker；主题可拥有多个分区以处理任意规模的数据。其次，分区是并行与负载均衡的单元——后文会结合消费者组介绍。

### Distributed

日志的分区分布在 Kafka 集群的多台服务器上，每台服务器保存数据并响应对分区的读写请求。每个分区会在多台服务器上进行副本备份以便容错，副本的数量（replication.factor）可配置。

每个分区由一个 leader 副本对外提供读写服务，其余为 follower。follower 采用复制机制被动同步 leader 的数据（in-sync replicas, ISR）。当 leader 故障时，ISR 中的一个副本会被自动选为新的 leader。集群内各 broker 既可作为某些分区的 leader，也可作为其他分区的 follower，以达到负载均衡。

### Producer

生产者（producer）可以将消息发布到某个主题（topic）。生产者可选择将消息发送到主题下的某个分区（partition），可采用轮询以均衡负载，也可按 key 进行一致性哈希以保证同 key 的消息落到同一分区（从而保证分区内顺序）。

### Consumer

传统消息系统有两种模型：队列与发布-订阅。队列中多消费者共享消费，一条消息只被一个消费者接收；发布-订阅中，一条消息广播给所有订阅者。Kafka 以“消费者组（consumer group）”统一抽象这两种模型。

消费者属于某个组，发布到主题的每条消息只会投递给订阅该主题的消费者组中的一个消费者实例（在该组内实现“每分区仅一个消费者”的并行消费）。如果所有消费者都在同一组，则等价于队列；如果每个消费者是独立组，则等价于发布-订阅。实际中常见的是：一个主题对应多个消费者组，每个组多个实例以实现扩展性与容错性。

Kafka 处理有序性的方式也和传统系统不同。传统队列允许任何消费者实例拉取消息，导致跨实例的顺序打散；Kafka 通过分区与“分区在组内只分配给一个实例”的约束，保证了**分区内有序**与**跨实例的负载均衡**。需要注意的是，**消费者实例数不能超过分区数**，否则会有空闲实例。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/1409263.png)

A two server Kafka cluster hosting four partitions (P0-P3) with two consumer groups. Consumer group A has two consumer instances and group B has four.

### 消息投递与一致性保证

1. 同一生产者发送到某主题分区的消息按发送顺序追加（append），即先进先出（FIFO）且 offset 单调递增；
2. 消费者实例按日志中的顺序读取消息（分区内有序保证成立，但跨分区不保证全局顺序）；
3. 对于副本因子为 N 的主题，在满足 in-sync 条件下，可容忍 N-1 台服务器故障而保证“已提交”的消息不丢失（需结合 acks=all 与 min.insync.replicas）。

### 典型应用场景

[http://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying](http://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying)

---

### 进一步说明与实践建议

- 副本与一致性：
  - `acks` 选择：`acks=0/1/all` 分别对应“最多一次/至少一次/更强一致”的生产语义；生产保障常用 `acks=all + min.insync.replicas>=2`；
  - ISR 与高水位（HW）：仅当消息被 ISR 多数副本确认，才对消费者“可见”；避免开启不安全的 ULE（unclean leader election）。
  - 精确一次（Exactly-Once, EOS）：启用幂等生产（`enable.idempotence=true`）+ 事务（`transactional.id`）可在 Kafka 内部实现端到端 EOS；Kafka Streams 对 EOS 提供一键支持。

- 保留策略与压缩：
  - 保留按时间（`retention.ms`）或大小（`retention.bytes`），压缩主题（log compaction）按 key 保留最新版本，适合维护“变更日志/物化视图”。
  - 消息压缩（`compression.type`）推荐 `lz4/zstd`，充分利用批处理（`batch.size`、`linger.ms`）提升吞吐与压缩比。

- 生产者与分区：
  - 分区选择：带 key 的一致性分区可保证“同 key 有序”；无 key 轮询均衡吞吐；
  - 吞吐优化：批量与 linger、压缩、合理的 `max.in.flight.requests.per.connection` 与 `retries`/`delivery.timeout.ms` 配合幂等保障。

- 消费者与偏移：
  - 偏移存储在内部主题 `__consumer_offsets` 中；提交策略包括自动/手动同步或异步提交；
  - 组重平衡（Rebalance）：Sticky/Range/RoundRobin 分配策略；心跳与 `max.poll.interval.ms` 影响会话保活与阻塞检测。

- 运维观测：
  - 监控关键指标：topic/partition 的 lag、ISR 大小波动、under-replicated partitions、controller 选举次数、磁盘/网络饱和；
  - 分区与副本：生产环境常见基线 `replication.factor=3`，`min.insync.replicas=2`，分区数与消费者实例数协调，避免过多小分区带来的元数据与文件句柄压力。

- 集群架构：
  - 早期版本依赖 ZooKeeper；自 2.8/3.0 起引入 KRaft 模式（无 ZooKeeper），以内置共识管理元数据与 controller 角色，简化部署与伸缩。

- 生态工具：
  - Kafka Connect（Source/Sink 连接器）用于对接外部系统；
  - Kafka Streams/Flink/Spark Streaming 用于有状态流处理与一致性保障（如 EOS）。

#### 参考资料（精选）
- Kafka 官方文档（Producer/Consumer/Design）：https://kafka.apache.org/documentation/
- Kafka Improvement Proposals（KIP，含 KRaft/EOS 等）：https://cwiki.apache.org/confluence/display/KAFKA/Kafka+Improvement+Proposals
- Exactly-Once Semantics in Kafka：https://kafka.apache.org/documentation/#semantics_eos
- Log Compaction 说明：https://kafka.apache.org/documentation/#compaction
- Kafka Protocol（基于 TCP 的二进制协议）：https://kafka.apache.org/protocol
- Kafka Streams 文档：https://kafka.apache.org/documentation/streams/




