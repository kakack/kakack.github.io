---

layout: post
categories: [Hadoop]
tags: [Big Data, Hadoop, Distributed System, HBase]

---

Hbase是HDFS上面向类列的分布式数据库，适合实时随机访问超大规模数据集。所有储存内容都以字符串格式存在，行中的列被分为“列族”（Column Family），同一个列族的所有列成员都有相同的前缀。一个表的所有列族都必须在创建（模式定义）的时候完成定义，但是新的列族成员可以后续不断加入。在物理上，同列族的成员都存放在同一个文件系统中。

##Feature

- Linear and modular scalability.

- Strictly consistent reads and writes.

- Automatic and configurable sharding of tables

- Automatic failover support between RegionServers.

- Convenient base classes for backing Hadoop MapReduce jobs with Apache HBase tables.

- Easy to use Java API for client access.

- Block cache and Bloom Filters for real-time queries.

- Query predicate push down via server side Filters

- Thrift gateway and a REST-ful Web service that supports XML, Protobuf, and binary data encoding options

- Extensible jruby-based (JIRB) shell

- Support for exporting metrics via the Hadoop metrics subsystem to files or Ganglia; or via JMX

##Terminology
### Row Key

与nosql数据库们一样,row key是用来检索记录的主键。访问hbase table中的行，只有三种方式：

1.  通过单个row key访问

2. 通过row key的range

3. 全表扫描

存储时，数据按照Row key的字典序(byte order)排序存储。设计key时，要充分排序存储这个特性，将经常一起读取的行存储放到一起(位置相关性)。行的一次读写是原子性的。

###列族

每个列都归属于某个列族，列族是表的schema的一部分，必须在使用之前定义好。访问控制、磁盘和内存的使用统计都是在列族层面进行的。实际应用中，列族上的控制权限能帮助我们管理不同类型的应用：我们允许一些应用可以添加新的基本数据、一些应用可以读取基本数据并创建继承的列族、一些应用则只允许浏览数据（甚至可能因为隐私的原因不能浏览所有数据）。所有权限的控制都发生在列族这一层。

###单元 Cell

HBase中通过row和columns确定的为一个存贮单元称为cell。由{row key, column( =<family> + <label>), version} 唯一确定的单元。cell中的数据是没有类型的，全部是字节码形式存贮。

###时间戳 TimeStamp

每个cell都保存着同一份数据的多个版本。版本通过时间戳来索引。时间戳的类型是 64位整型。时间戳可以由hbase(在数据写入时自动 )赋值，此时时间戳是精确到毫秒的当前系统时间。时间戳也可以由客户显式赋值。如果应用程序要避免数据版本冲突，就必须自己生成具有唯一性的时间戳。每个cell中，不同版本的数据按照时间倒序排序，即最新的数据排在最前面。

- - -

##HBase接口

方式|特点|场合
--------|------|-----
Native Java API|最常规和高效|Hadoop MapReduce Job并行处理HBase表数据
HBase Shell|最简单接口|HBase管理使用
Thrift Gateway|利用Thrift序列化支持多种语言|异构系统在线访问HBase表数据
Rest Gateway|解除语言限制	|Rest风格Http API访问
Pig|Pig Latin六十编程语言处理数据|数据统计
Hive|简单，SqlLike|

- - -

##Table&Region

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/16113.jpg)

1. Table随着记录增多不断变大，会自动分裂成多份Splits，成为Regions
2. 一个region由[startkey，endkey]表示
3. 不同region会被Master分配给相应的RegionServer进行管理

- - -

##-ROOT- & .META.

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161132.jpg)

- .META. 　　记录用户表的Region信息，同时，.META.也可以有多个region
- -ROOT-	　  记录.META.表的Region信息，但是，-ROOT-只有一个region

Zookeeper中记录了-ROOT-表的location
客户端访问数据的流程：
Client -> Zookeeper -> -ROOT- -> .META. -> 用户数据表
多次网络操作，不过client端有cache缓存

- - -

##HBase系统架构图

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161133.jpg)

###Client：
使用HBase RPC机制与HMaster和HRegionServer进行通信
Client与HMaster进行通信进行管理类操作
Client与HRegionServer进行数据读写类操作

###Zookeeper：
Zookeeper Quorum存储-ROOT-表地址、HMaster地址
HRegionServer把自己以Ephedral方式注册到Zookeeper中，HMaster随时感知各个HRegionServer的健康状况
Zookeeper避免HMaster单点问题

###HMaster：
HMaster没有单点问题，HBase中可以启动多个HMaster，通过Zookeeper的Master Election机制保证总有一个Master在运行
主要负责Table和Region的管理工作：

1. 管理用户对表的增删改查操作
2. 管理HRegionServer的负载均衡，调整Region分布
3. Region Split后，负责新Region的分布
4. 在HRegionServer停机后，负责失效HRegionServer上Region迁移

###HRegionServer：
HBase中最核心的模块，主要负责响应用户I/O请求，向HDFS文件系统中读写数据

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161134.jpg)

HRegionServer管理一些列HRegion对象；
每个HRegion对应Table中一个Region，HRegion由多个HStore组成；
每个HStore对应Table中一个Column Family的存储；
Column Family就是一个集中的存储单元，故将具有相同IO特性的Column放在一个Column Family会更高效

###HStore：
HBase存储的核心。由MemStore和StoreFile组成。
MemStore是Sorted Memory Buffer。用户写入数据的流程：

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161135.gif)

Client写入 -> 存入MemStore，一直到MemStore满 -> Flush成一个StoreFile，直至增长到一定阈值 -> 出发Compact合并操作 -> 多个StoreFile合并成一个StoreFile，同时进行版本合并和数据删除 -> 当StoreFiles Compact后，逐步形成越来越大的StoreFile -> 单个StoreFile大小超过一定阈值后，触发Split操作，把当前Region Split成2个Region，Region会下线，新Split出的2个孩子Region会被HMaster分配到相应的HRegionServer上，使得原先1个Region的压力得以分流到2个Region上
由此过程可知，HBase只是增加数据，有所得更新和删除操作，都是在Compact阶段做的，所以，用户写操作只需要进入到内存即可立即返回，从而保证I/O高性能。

###HLog
引入HLog原因：
在分布式系统环境中，无法避免系统出错或者宕机，一旦HRegionServer以外退出，MemStore中的内存数据就会丢失，引入HLog就是防止这种情况
工作机制：
每个HRegionServer中都会有一个HLog对象，HLog是一个实现Write Ahead Log的类，每次用户操作写入Memstore的同时，也会写一份数据到HLog文件，HLog文件定期会滚动出新，并删除旧的文件(已持久化到StoreFile中的数据)。当HRegionServer意外终止后，HMaster会通过Zookeeper感知，HMaster首先处理遗留的HLog文件，将不同region的log数据拆分，分别放到相应region目录下，然后再将失效的region重新分配，领取到这些region的HRegionServer在Load Region的过程中，会发现有历史HLog需要处理，因此会Replay HLog中的数据到MemStore中，然后flush到StoreFiles，完成数据恢复。

###HBase存储格式
HBase中的所有数据文件都存储在Hadoop HDFS文件系统上，格式主要有两种：
1 HFile HBase中KeyValue数据的存储格式，HFile是Hadoop的二进制格式文件，实际上StoreFile就是对HFile做了轻量级包装，即StoreFile底层就是HFile
2 HLog File，HBase中WAL（Write Ahead Log） 的存储格式，物理上是Hadoop的Sequence File

###HFile

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161136.jpg)

HFile文件不定长，长度固定的块只有两个：Trailer和FileInfo
Trailer中指针指向其他数据块的起始点
File Info中记录了文件的一些Meta信息，例如：AVG_KEY_LEN, AVG_VALUE_LEN, LAST_KEY, COMPARATOR, MAX_SEQ_ID_KEY等
Data Index和Meta Index块记录了每个Data块和Meta块的起始点
Data Block是HBase I/O的基本单元，为了提高效率，HRegionServer中有基于LRU的Block Cache机制
每个Data块的大小可以在创建一个Table的时候通过参数指定，大号的Block有利于顺序Scan，小号Block利于随机查询
每个Data块除了开头的Magic以外就是一个个KeyValue对拼接而成, Magic内容就是一些随机数字，目的是防止数据损坏

HFile里面的每个KeyValue对就是一个简单的byte数组。这个byte数组里面包含了很多项，并且有固定的结构。

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161137.jpg)

KeyLength和ValueLength：两个固定的长度，分别代表Key和Value的长度
Key部分：Row Length是固定长度的数值，表示RowKey的长度，Row 就是RowKey
Column Family Length是固定长度的数值，表示Family的长度
接着就是Column Family，再接着是Qualifier，然后是两个固定长度的数值，表示Time Stamp和Key Type（Put/Delete）
Value部分没有这么复杂的结构，就是纯粹的二进制数据

###HLog File

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161138.jpg)

HLog文件就是一个普通的Hadoop Sequence File，Sequence File 的Key是HLogKey对象，HLogKey中记录了写入数据的归属信息，除了table和region名字外，同时还包括 sequence number和timestamp，timestamp是“写入时间”，sequence number的起始值为0，或者是最近一次存入文件系统中sequence number。
HLog Sequece File的Value是HBase的KeyValue对象，即对应HFile中的KeyValue

- - -

##Shell指令

1, 进入shell端。

```
$ ./bin/hbase shell
hbase(main):001:0>
```

2, 帮助。`help`

3, 创建新表。

```
hbase(main):001:0> create 'test', 'info'
0 row(s) in 0.4170 seconds

=> Hbase::Table - test
```

4, 显示现有的所有表名

```
hbase(main):002:0> list
TABLE
test
test_info
2 row(s) in 4.1590 seconds

=> ["test", "test_info"]
```

5, 描述某个表详细结构

```
hbase(main):005:0> describe 'test'
DESCRIPTION                                                              ENABLED
 'test', {NAME => 'info', DATA_BLOCK_ENCODING => 'NONE', BLOOMFILTER =>  true
 'ROW', REPLICATION_SCOPE => '0', VERSIONS => '1', COMPRESSION => 'NONE'
 , MIN_VERSIONS => '0', TTL => 'FOREVER', KEEP_DELETED_CELLS => 'false',
  BLOCKSIZE => '65536', IN_MEMORY => 'false', BLOCKCACHE => 'true'}
1 row(s) in 0.0790 seconds

```

6, 往某个表中添加数据

```
hbase(main):014:0> put 'test', 'testRow1', 'info:a','aaa'
0 row(s) in 0.2790 seconds

hbase(main):015:0> put 'test', 'testRow2', 'info:a','a22'
0 row(s) in 0.0350 seconds

hbase(main):016:0> put 'test', 'testRow1', 'info:b','bbb'
0 row(s) in 0.0100 seconds

```

以第一条指令为例，‘test’确定数据库表，‘testRow1’和‘info:a’确定行列，在此，a是‘info’这个column family中的成员， ‘aaa’是实际插入的数据值。

7，扫描整个表单

```
hbase(main):020:0> scan 'test'
ROW                           COLUMN+CELL
 testRow1                     column=info:a, timestamp=1452624429698, value=aaa
 testRow1                     column=info:b, timestamp=1452624455246, value=bbb
 testRow2                     column=info:a, timestamp=1452624444189, value=a22
2 row(s) in 0.0400 seconds

```

8, 获取单行数据

```
hbase(main):022:0> get 'test', 'testRow1'
COLUMN                        CELL
 info:a                       timestamp=1452624429698, value=aaa
 info:b                       timestamp=1452624455246, value=bbb
2 row(s) in 0.0390 seconds

```

9，将某张数据库表失效，在删除或做其他类似操作之前，必须将某张表做disable操作。

```
hbase(main):029:0> disable 'scores_3'
0 row(s) in 1.2850 seconds

```

10，删除某个表

```
hbase(main):030:0> drop 'scores_3'
0 row(s) in 1.0390 seconds

```



