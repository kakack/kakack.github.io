---

layout: post
categories: [Spark]
tags: [Spark,Big Data,Distributed System]

---

# Spark Programming Guide

### 概述

整体上，每个Spark应用都是由*driver program*组成的，它们用来运行用户的main function，并且在一个cluster上执行多种并行的操作。Spark提供的主要的抽象化结果是resilient distributed dataset (RDD：弹性分布式数据集) （可以详见paper：resilient distributed dataset：A fault-Tolerant Abstraction for In-Memory Cluster Computing）。RDD可以通过启动已有的HDFS上的文件来创建，也可以通过一个已存在的driver program上的Scala集合，然后转化它。用户同样可以要求Spark在内存中保持一个RDD，允许它被重用。最后，RDD可以从node failure中自动被重用。

另一个抽象产物是shared variables，可以并行操作。当在一组nodes上并行运行任务的时候，Spark会把function中每个变量的拷贝都传递出去。有的时候，一个变量需要在任务之间或者任务和driver program之间被共用和传递。Spark支持两种shared variables：broadcast variables，可以在所有nodes的内存中用来缓存一个值；以及accumulators，只能用来加操作，比如计数器和求和

### 连接Spark

Spark1.0.2用的是Scala2.10的库，如果用Maven，那么Spark可以用

    groupId = org.apache.spark
    artifactId = spark-core_2.10
    version = 1.0.2
    
如果想访问一个HDFS cluster，需要添加一个hadoop-client的依赖
   
    groupId = org.apache.hadoop
    artifactId = hadoop-client
    version = <your-hdfs-version>
    
最后，需要导入Spark的包和库文件

    import org.apache.spark.SparkContext
    import org.apache.spark.SparkContext._
    import org.apache.spark.SparkConf
    
### 初始化Spark

Spark Program第一件事就是创建一个SparkContext object，用来告诉Spark如何访问一个cluster，为了创建一个SparkContex，你首先要建立一个SparkConf object，其中包含了你的应用的信息

    val conf = new SparkConf().setAppName(appName).setMaster(master)
    new SparkContext(conf)
    
##### 使用Shell

在Spark-shell中，一种特殊的交互SparkContext 已经被默认建立好了，在变量sc中。可以自行决定关联的master， 用--master，可以添加jar，用--jar，比如

`$ ./bin/spark-shell --master local[4]`

或者添加code.jar到这个classpath里去

`$ ./bin/spark-shell --master local[4] --jars code.jar`

### 关于Resilient Distributed Datasets (RDDs)

这部分等我看了paper之后再来慢慢整理，可详见我之后关于RDD的介绍

### Shared Variables

通常，当一个function传递到一个Spark操作（如mao或者reduce），并且在一个远程的cluster node上被执行了，它会分别在每个变量copy上被使用到。这些变量被复制到各个机器上，而且这些变量如果在远程机器上得到更新，是不会传递回driver program上的。如果在tasks之间支持通用的，可供读写的shared variables，显然会造成低效，所以Spark提供了两种类型有限的shared variables：*Broadcast Variables*和*accumulators*

##### Broadcast Variables

Broadcast Variables允许程序员保持一个只读变量在各个机器的缓存里，而不是通过tasks分配变量的copy。它们可以用来非常高效地给每一个node提供一份大规模input dataset的copy，Spark同样也尝试使用高效的广播算法（Broadcast Algorithm）去分布broadcast variables，从而减少通讯开销。

一个从变量v中创建出来的broadcast variable，是通过调用`SparkContext.broadcast(v)`，这个broadcast variable是v的一个封装，它的值可以通过调用`value`方法访问到。

        scala> val broadcastVar = sc.broadcast(Array(1, 2, 3))
        broadcastVar: spark.Broadcast[Array[Int]] = spark.Broadcast(b5c40191-a864-4c7d-b9bf-d87e1a4e787c)
        
        scala> broadcastVar.value
        res0: Array[Int] = Array(1, 2, 3)

在Broadcast Variables被创建出来后，它应该用来在所有cluster上的function中代替原先的v，所以v不会被重复复制分配到各个nodes上。此外，对象v在确认它被广播到所有nodes前，不应该被修改。

##### Accumulators

Accumulators是只用来做“加”操作的，因此在并行操作中很容易被支持，可以用于计数器和求和中，Spark默认支持的是numeric type，程序员可以自行添加支持的类型。

一个accumlator通过调用`SparkContext.accumulator(v)`从最初值v中创建，然后在cluster上运行的tasks可以用`add`方法或者+=操作符来在它上面做加运算。但是不能读取这个数值，只有driver program能够读取这个值，通过其`value`方法

        scala> val accum = sc.accumulator(0)
        accum: spark.Accumulator[Int] = 0

        scala> sc.parallelize(Array(1, 2, 3, 4)).foreach(x => accum += x)
        ...
        10/09/29 18:41:08 INFO SparkContext: Tasks finished in 0.317106 s

        scala> accum.value
        res2: Int = 10

上述代码用的是build-in support的Int类型，如果想加入自定义的类型，可以在`AccumulatorParam`下创建子类。AccumulatorParam interface提供两个方法：`zero`为新建类型提供“zero value”，而`addInPlace`用来把两个值加起来，例如

        object VectorAccumulatorParam extends AccumulatorParam[Vector] {
          def zero(initialValue: Vector): Vector = {
            Vector.zeros(initialValue.size)
          }
          def addInPlace(v1: Vector, v2: Vector): Vector = {
            v1 += v2
          }
        }

        // Then, create an Accumulator of this type:
        val vecAccum = sc.accumulator(new Vector(...))(VectorAccumulatorParam)


### 部署到集群上

在[application submission guide](http://spark.apache.org/docs/latest/submitting-applications.html)上完整描述了如何部署Spark应用到一个cluster上，总而言之，一旦你把自己的应用打包好刀一个JAR（给java、scala），或者到一组.py或者.zip文件中，脚本` bin/spark-submit `会让你确认提交它到任何一个支持的cluster manager上

### 单元测试

Spark对于单元测试非常友好，简单来说可以创建一个SparkContext在自己的测试中，用`local`代替master URL，运行你的Spark应用然后用`SparkContext.stop()`来叫停，保证你contex有一个`finally`块或者是测试框架的`tearDown`方法，因为Spark不支持两个contex同时运行在一个程序上。

### 参考文档

 - [Some example Spark programs ](http://spark.apache.org/examples.html)

可以用`./bin/run-example SparkPi`来运行例子。







