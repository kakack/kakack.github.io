---

layout: post
categories: [Spark]
tags: [Big Data, Apache Kafka, Distributed System, Spark Streaming]

---

详细Demo例子与代码位于[kafka-storm-starter](https://github.com/miguno/kafka-storm-starter)，其中kafka与Spark Streaming的嵌合是其中一个test case。

Test Case名称为：KafkaSparkStreamingSpec。测试内容是，开一个Streaming Job从Kafka读入输入数据并写回到Kafka。展示了如何并行地从一个topic的所有partitions读取输入数据，如何通过partition的编号来解耦downstream parallelism，如何将数据重新写回到kafka。这里的输入输出数据都是以Arvo格式存在，我们使用Twitter Bijection来做序列化工作。

- - - 

在Kafka中，所有消息数据都会在topic中做中转或者说缓存，每个topic中会配置多个分区partitions，这个值一般都是消费者parallelism的最大数目，比如一个topic开了5个分区，那么最大支持5个线程并行进行消费。这5个线程，也可以理解为消费者群(consumer group)，即用所选择的名称进行识别来区分消费者应用程序集群范围。

举个例子：如果有一个叫“A”的消费者群从一个叫“B”的Kafka topic进行读取，这个topic有10个分区。如果“A”开1个线程，那么会从10个分区中分别读取，如果开5个线程，那么每个线程从2个分区进行读取，如果开10个那就每个线程一个分区，如果开20个，那就其中10个线程每个线程一个分区另外10个线程空闲。如果这个提升过程是逐步的，即从一个线程慢慢增加到20个，那么kafka自己会有一个再平衡的过程，会逐步将topic中的分区以绝对公平地原则分配给各个线程。

- Read Parallelism：通常会期望最大化利用kafka topic中的分区，即用N个线程并行处理N个分区，并希望这些线程能跨主机工作。

- Downstream processing parallelism：相对于读取数据的线程，希望有更多的线程来对数据进行处理，可以通过多个读取线程shuffling来实现。

- - -
##读取

Spark Streaming中的KafkaInputDStream（kafka连接器）使用了kafka的[high-level consumer API](https://cwiki.apache.org/confluence/display/KAFKA/Consumer+Group+Example)，意味这这里用两种办法能控制Spark从kafka读取的parallelism：

1. Input DStream的数量：每个input DStream上都会有个一个receiver(=task)，即可以使用多个input DStream跨多个节点进行并行操作。
2. 每个input DStream上consumer threads的数目：相同的receiver(=task)上将会运行多个读取线程，读取操作在每个主机上并行进行。

由于受到网络传输/NIC的影响，事实上在同一个主机上即便运行多个线程也不一定能明显增加吞吐量，另外从kafka中读取也会受到cpu瓶颈的限制。另外，多个读取线程在推送数据到block的时候，有可能会产生锁竞争，所以方法1是更好的选择。

######控制input DStream的实践：

```
val ssc: StreamingContext = ……
// 创建SSC

val kafkaParams: Map[String, String] = 
Map("group.id" -> "terran", /* ignore rest */)

val numInputDStreams = 5
//创建5个Input DStream

val kafkaDStreams = (1 to numInputDStreams).map { _ => KafkaUtils.createStream(...) }

```

######控制每个input DStream上小线程的数目

这个例子中只有一个input DStream，但有3个消费者线程

```
val ssc: StreamingContext = ???
 // ignore for now

val kafkaParams: Map[String, String] = 
Map("group.id" -> "terran", ...)

val consumerThreadsPerInputDstream = 3
//设置每个Input DStream上的consumer线程数为3

val topics = Map("zerg.hydra" -> consumerThreadsPerInputDstream)
val stream = KafkaUtils.createStream(ssc, kafkaParams, topics, ...)


```

######结合起来

```
val ssc: StreamingContext = ???
val kafkaParams: Map[String, String] = Map("group.id" -> "terran", ...)

val numDStreams = 5
val topics = Map("zerg.hydra" -> 3)
//设置5个DStream，每个上跑3消费者个线程

val kafkaDStreams = (1 to numDStreams).map { _ =>
    KafkaUtils.createStream(ssc, kafkaParams, topics, ...)
  }
```

- - -

##对并行Downstream的处理

Spark环境下本身是用RDD来处理并行化的任务的，因此Kafka中的parallelism跟RDD的数目有关。控制的方法也有两个：

1. Input DStream的数目：同上
2. DStream transformation的重分配：这里将获得一个全新的DStream，其parallelism等级可能增加、减少，或者保持原样。在DStream中，每个返回的RDD都恰好有N个分区，DStream就是由一连串连续的RDD组成的，在这一场景中，`DStream.repartition`通过`RDD.repartition`实现，接下来将对RDD中的所有数据做随机的reshuffle。

总的来说repartition是从processing parallelism解耦read parallelism的主要途径。在这个过程中一个关键的transformation是union操作，这个操作会将多个DStream压缩到一个DStream或者RDD当中。例如可以在5 read parallelism的topic中，将processing parallelism提升到20：

```
val ssc: StreamingContext = ???
val kafkaParams: Map[String, String] = Map("group.id" -> "terran", ...)
val readParallelism = 5
val topics = Map("zerg.hydra" -> 1)

val kafkaDStreams = (1 to readParallelism).map { _ =>
    KafkaUtils.createStream(ssc, kafkaParams, topics, ...)
  }
//> collection of five *input* DStreams = handled by five receivers/tasks，同上

val unionDStream = ssc.union(kafkaDStreams) 
// often unnecessary, just showcasing how to do it
//> single DStream

val processingParallelism = 20
val processingDStream = unionDStream(processingParallelism)
//> single DStream but now with 20 partitions

```

- - -

##写入到Kafka

写入到Kafka需要从foreachRDD输出操作进行，通用的输出操作往往让每个RDD都由DStream生成，这个函数需要将每个RDD中的数据推送到一个外部系统，比如保存到文件系统，或者写入数据库，或者通过socket发送等。需要注意的是，这里的功能函数将在驱动中执行，同时其中通常会伴随RDD行为，它将会促使流RDDs的计算。

在本例子中，使用的办法是横跨多个RDD/batch，通过一个producer pool来重用kafka生产者实例。实现了接口[Apache Common Pool](http://commons.apache.org/proper/commons-pool/)，详见[PooledKafkaProducerAppFactory](https://github.com/miguno/kafka-storm-starter/blob/develop/src/main/scala/com/miguno/kafkastorm/kafka/PooledKafkaProducerAppFactory.scala)。

如下：

```
val producerPool = {
  // See the full code on GitHub for details on how the pool is created
  val pool = createKafkaProducerPool(kafkaZkCluster.kafka.brokerList, outputTopic.name)
  ssc.sparkContext.broadcast(pool)
}

stream.map { ... }.foreachRDD(rdd => {
  rdd.foreachPartition(partitionOfRecords => {
    // Get a producer from the shared pool
    val p = producerPool.value.borrowObject()
    partitionOfRecords.foreach { case tweet: Tweet =>
      // Convert pojo back into Avro binary format
      val bytes = converter.value.apply(tweet)
      // Send the bytes to Kafka
      p.send(bytes)
    }
    // Returning the producer to the pool also shuts it down
    producerPool.value.returnObject(p)
  })
})
```

Spark Streaming每分钟都会建立起多个RDD，每个都会包括很多partition，所以用户无需为每个partition上建立新的kafka生产者或者kafka message。这个producer pool能够精确控制应用中的kafka生产者实例数量。

- - -

##完整示例

流程：

1. 并行地从Kafka topic中读取Avro-encoded数据，每个Kafka partition都配置了一个单线程 input DStream。
2. 并行化Avro-encoded数据到pojos中，然后将他们并行写到binary，序列化可以通过 Twitter Bijection执行。
3. 通过Kafka生产者池将结果写回一个不同的Kafka topic。

```
// Set up the input DStream to read from Kafka (in parallel)
val kafkaStream = {
  val sparkStreamingConsumerGroup = "spark-streaming-consumer-group"
  val kafkaParams = Map(
    "zookeeper.connect" -> "zookeeper1:2181",
    "group.id" -> "spark-streaming-test",
    "zookeeper.connection.timeout.ms" -> "1000")
  val inputTopic = "input-topic"
  val numPartitionsOfInputTopic = 5
  val streams = (1 to numPartitionsOfInputTopic) map { _ =>
    KafkaUtils.createStream(ssc, kafkaParams, Map(inputTopic -> 1), StorageLevel.MEMORY_ONLY_SER).map(_._2)
  }
  val unifiedStream = ssc.union(streams)
  val sparkProcessingParallelism = 1 // You'd probably pick a higher value than 1 in production.
  unifiedStream.repartition(sparkProcessingParallelism)
}

// We use accumulators to track global "counters" across the tasks of our streaming app
val numInputMessages = ssc.sparkContext.accumulator(0L, "Kafka messages consumed")
val numOutputMessages = ssc.sparkContext.accumulator(0L, "Kafka messages produced")
// We use a broadcast variable to share a pool of Kafka producers, which we use to write data from Spark to Kafka.
val producerPool = {
  val pool = createKafkaProducerPool(kafkaZkCluster.kafka.brokerList, outputTopic.name)
  ssc.sparkContext.broadcast(pool)
}
// We also use a broadcast variable for our Avro Injection (Twitter Bijection)
val converter = ssc.sparkContext.broadcast(SpecificAvroCodecs.toBinary[Tweet])

// Define the actual data flow of the streaming job
kafkaStream.map { case bytes =>
  numInputMessages += 1
  // Convert Avro binary data to pojo
  converter.value.invert(bytes) match {
    case Success(tweet) => tweet
    case Failure(e) => // ignore if the conversion failed
  }
}.foreachRDD(rdd => {
  rdd.foreachPartition(partitionOfRecords => {
    val p = producerPool.value.borrowObject()
    partitionOfRecords.foreach { case tweet: Tweet =>
      // Convert pojo back into Avro binary format
      val bytes = converter.value.apply(tweet)
      // Send the bytes to Kafka
      p.send(bytes)
      numOutputMessages += 1
    }
    producerPool.value.returnObject(p)
  })
})

// Run the streaming job
ssc.start()
ssc.awaitTermination()

```

[完整源码](https://github.com/miguno/kafka-storm-starter/blob/develop/src/test/scala/com/miguno/kafkastorm/spark/KafkaSparkStreamingSpec.scala)

- - -

##终端测试输出

![](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/Screen%20Shot%202015-11-04%20at%2010.39.04.png)
