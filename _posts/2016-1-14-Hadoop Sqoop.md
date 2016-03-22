---

layout: post
categories: [Hadoop]
tags: [Big Data, Hadoop, Distributed System, Sqoop]

---

Sqoop 是 apache 下用于 RDBMS 和 HDFS 互相导数据的工具。安装过程懒得说了。主要use case是从关系型数据库中把数据导入到HDFS上。

##架构

Sqoop的架构很简单，如下：

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/161141.jpg)

其整合了Hive、Hbase和Oozie，通过map-reduce任务来传输数据，从而提供并发特性和容错。sqoop主要通过JDBC和关系数据库进行交互。理论上支持JDBC的database都可以使用sqoop和hdfs进行数据交互。

- - -

##可用命令

Available commands:


- **codegen**：Generate code to interact with database records
- **create-hive-table**：Import a table definition into Hive
- **eval**：Evaluate a SQL statement and display the results
- **export**：Export an HDFS directory to a database table
- **help**：List available commands
- **import**：Import a table from a database to HDFS
- **import-all-tables**：Import tables from a database to HDFS
- **import-mainframe**：Import datasets from a mainframe server to HDFS
- **job**：Work with saved jobs
- **list-databases**：List available databases on a server
- **list-tables**：List available tables in a database
- **merge**：Merge results of incremental imports
- **metastore**：Run a standalone Sqoop metastore
- **version**：Display version information

- - -

##Import操作

用的最多的一条命令。数据导入有以下一些特点：

1. 支持文本文件(--as-textfile)、avro(--as-avrodatafile)、SequenceFiles(--as-sequencefile)。 RCFILE暂未支持，默认为文本
2. 支持数据追加，通过--apend指定
3. 支持table列选取（--column），支持数据选取（--where），和--table一起使用
4. 支持数据选取，例如读入多表join后的数据'SELECT a.*, b.* FROM a JOIN b on (a.id == b.id) ‘，不可以和--table同时使用
5. 支持map数定制(-m)
6. 支持压缩(--compress)
7. 支持将关系数据库中的数据导入到Hive(--hive-import)、HBase(--hbase-table)


Sqoop在import时，需要制定split-by参数。Sqoop根据不同的split-by参数值来进行切分,然后将切分出来的区域分配到不同map中。每个map中再处理数据库中获取的一行一行的值，写入到HDFS中。同时split-by根据不同的参数类型有不同的切分方法，如比较简单的int型，Sqoop会取最大和最小split-by字段值，然后根据传入的num-mappers来确定划分几个区域。 比如select max(split_by),min(split-by) from得到的max(split-by)和min(split-by)分别为1000和1，而num-mappers为2的话，则会分成两个区域(1,500)和(501-100),同时也会分成2个sql给2个map去进行导入操作，分别为select XXX from table where split-by>=1 and split-by<500和select XXX from table where split-by>=501 and split-by<=1000。最后每个map各自获取各自SQL中的数据进行导入工作。

- - -

##例子

源数据库地址：`jdbc:db2://10.37.146.111:5912/c99`

账号密码：`sapcp1:1qaz2wsx`

待导入的表名：`source`

导入后的表名：`bw_test.dest`

导入表结构到hive中```sqoop create-hive-table
--connect jdbc:db2://10.37.146.111:5912/c99--table source --username sapcp1 --password 1qaz2wsx --hive-table bw_test.dest;```同时导出数据和表结构```sqoop import --connect jdbc:db2://10.37.146.111:5912/c99 --username sapcp1 --password 1qaz2wsx --table source --columns "ANLAGE,BRANCHE" --fields-terminated-by "\t"   //行分割符--lines-terminated-by "\n"   //列分割符--hive-import --hive-overwrite --create-hive-table --hive-table bw_test.dest --delete-target-dir```导入表到HDFS上```sqoop import  
--connect jdbc:db2://10.37.146.111:5912/c99  --username sapcp1 --password 1qaz2wsx --query "select * from source where \$CONDITIONS "  -m 1  --target-dir /user/hive/warehouse/bw_test.db/dest--fields-terminated-by "\001" --delete-target-dir```列出数据库
```sqoop list-databases--connect jdbc:db2://10.37.146.111:5912 --username sapcp1 --password 1qaz2wsx```列出数据库表```sqoop-list-tables --connect jdbc:db2://10.37.146.111:5912/c99 --username sapcp1 --password 1qaz2wsx```

  
