---

layout: post
categories: [Cloudera]
tags: [Big Data, Hadoop, Distributed System, Cloudera]

---

#CDH


CDH一个对Apache Hadoop的集成环境的封装，可以使用Cloudera Manager进行自动化安装。相对于Apache而言，CDH对于Hadoop的版本标识划分地更清晰，如CDH3、CDH4、CDH5等。提供对多个操作系统的支持，可以用apt-get或者yum等命令进行安装和依赖下载。

- [文档](http://www.cloudera.com/documentation.html)
- [CDH5汇总](http://archive.cloudera.com/cdh5/)

主要功能：管理、监控、诊断、集成。

###主要组件：

- HTTPFS：HDFS的一个HTTP借口，通过WebHDFS REST API可以对HDFS进行读写访问，不需要客户端可以访问Hadoop每个节点，镶嵌在tomcat中
- HBase：建立在HDFS上的额列存储数据库。
- HDFS：适合在通用硬件上的分布式文件系统
- Hive：基于Hadoop的数据仓库工具，用于存储和处理海量结构化数据。
- Hue：Hue是CDH专门的一套WEB管理器，它包括3个部分Hue Ui，Hue Server，Hue db。Hue提供所有的CDH组件的Shell界面的接口。
- Impala：提供对HDFS和Hbase查询的SQL。
- MapReduce：变成模式，用于大规模数据集并行计算。
- Oozie：开源工作流引擎，协调运行在Hadoop平台上的Jobs。
- Solr：基于Lucene的Java搜索引擎服务器，提供层面搜索、命中醒目显示灯输出格式。
- Spark：略
- Sqoop：从关系型数据库传导数据到HDFS，通过JDBC与关系数据库交互。
- Yarn：理解为MapReduceV2版本，将资源管理和任务调度&监控两个功能分离成单独的组件，每一个应用的 ApplicationMaster 负责相应的调度和协调。
- Zookeeper：Apache的分布式服务框架，解决系统一致性的问题，能提供基于类似于文件系统的目录节点树方式的数据存储，但是 Zookeeper 并不是用来专门存储数据的，它的作用主要是用来维护和监控你存储的数据的状态变化。

- - -

#各个节点前期准备工作
 
###设置Hosts

在`/etc/hosts`中做修改，添加自身和其他子节点的ip、FQDN、hostname，如下：

```
*** System restart required ***

10.10.245.82 cloudera2.cci.com cloudera2

10.10.245.115 cloudera-node.cci.com cloudera-node

10.10.245.123 cloudera-node2.cci.com cloudera-node2
```

###设置Hostname

运行`su hostname cloudera-node2`

再在`/etc/hostname`里写入`cloudera-node2`

###关闭防火墙

关闭防火墙：`ufw disable`

卸载防火墙：`apt-get remove iptables`

###安装JDK（至少需要Oracle JDK7及以上）

```
java -version
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java7-installer
sudo apt-get install oracle-java7-set-default

```

通过命令`java -version`查看是否安装成功

然后添加JAVA_HOME和Path，在我这是在`~/.bashrc`里加入：

```
export JAVA_HOME=/usr/lib/jvm/java-7-oracle
export CLASSPATH=.:$JAVA_HOME/lib:$JAVA_HOME/jrelib:$CLASSPATH
export PATH=$JAVA_HOME/bin:$JAVA_HOME/jre/bin:$PATH

```

保存后`source ~/.bashrc`，通过命令`echo $JAVA_HOME`查看是否成功。

###安装Mysql并设置远程登录

`sudo apt-get install mysql-server`

修改`/etc/mysql/my.conf`，将`skip-external-locking`注释掉，并将`bind-address = 0.0.0.0`

重启MySQL 服务`/etc/init.d/mysql restart`

###安装ntp

`sudo apt-get install ntp`，重启服务`sudo service ntp restart`


- - -

#安装Cloudera Manager

###一，创建新的CDH5 Repository或者下载1-click Install Package
在Ubuntu上安装有两种不同的方法，一种是下载 "1-click Install" Package，一键安装，另一种是将CDH5 repository加入本地repo


#####通过1-click Install Package安装

下载Package：

OS Version | Package Link
-----------|-------------
Wheezy|[Wheezy Package](https://archive.cloudera.com/cdh5/one-click-install/wheezy/amd64/cdh5-repository_1.0_all.deb)
Precise|[Precise Package](https://archive.cloudera.com/cdh5/one-click-install/precise/amd64/cdh5-repository_1.0_all.deb)
Trusty|[Trusty Package](https://archive.cloudera.com/cdh5/one-click-install/trusty/amd64/cdh5-repository_1.0_all.deb)

运行`sudo dpkg -i cdh5-repository_1.0_all.deb`，在运行之前可以先做一下`apt-get update`

然后安装下载器：

`$ wget https://archive.cloudera.com/cm5/installer/latest/cloudera-manager-installer.bin`

更改访问权限：

`$ chmod u+x cloudera-manager-installer.bin`

运行Cloudera Manager Server安装器

- 如果通过网络的repo安装，则直接运行`$ sudo ./cloudera-manager-installer.bin`
- 如果通过本地repo安装，则运行`$ sudo ./cloudera-manager-installer.bin --skip_repo_package=1`

在安装完成之后，shell端会返回GUI对话框，所有都选accept、next，若无问题则可安装完成，运行一下`sudo service cloudera-scm-server start`

我个人建议用这种办法，因为下面提到的通过apt-get安装需要手动配置一些数据库相关的内容，比较繁琐。
#####添加CDH5 Repository

OS Version |Command
-----------|--------
Ubuntu Trusty (14.04)	|wget https://archive.cloudera.com/cm5/ubuntu/trusty/amd64/cm/cloudera.list -O /etc/apt/sources.list.d/cloudera.list
Ubuntu Precise (12.04)	|wget https://archive.cloudera.com/cm5/ubuntu/precise/amd64/cm/cloudera.list -O /etc/apt/sources.list.d/cloudera.list
Ubuntu Lucid (10.04)	|wget https://archive.cloudera.com/cm5/ubuntu/lucid/amd64/cm/cloudera.list -O /etc/apt/sources.list.d/cloudera.list


最后运行`apt-get update`，可能会返回因为没有repo key而不能fetch某些package，可以在完成下一步添加repo key之后，再运行一遍update命令 

*如果是Trusty的操作系统，还需要以下一个附加步骤：*

创建一个新文件`/etc/apt/preferences.d/cloudera.pref`

内如如下：

```
Package: *
Pin: release o=Cloudera, l=Cloudera
Pin-Priority: 501
```

###二，添加一个Repository Key

OS Version|	Command
----------|--------
Debian Wheezy|	$ wget https://archive.cloudera.com/cdh5/debian/wheezy/amd64/cdh/archive.key -O archive.key
Ubuntu Precise|	$ wget https://archive.cloudera.com/cdh5/ubuntu/precise/amd64/cdh/archive.key -O archive.key
Ubuntu Lucid	|$ wget https://archive.cloudera.com/cdh5/ubuntu/lucid/amd64/cdh/archive.key -O archive.key
Ubuntu Trusty|	$ wget https://archive.cloudera.com/cdh5/ubuntu/trusty/amd64/cdh/archive.key -O archive.key

然后运行  `$ sudo apt-key add archive.key`

###三，安装Yarn
这步可以选装，因为Yarn在之后的节点安装过程中也是自带的。

如果需要运行MapReduce V2，那么需要安装Yarn。

- 在Resource Manager host上运行：`sudo apt-get update; sudo apt-get install hadoop-yarn-resourcemanager`

- 在NameNode Host上运行`sudo apt-get install hadoop-hdfs-namenode`

- 在Secondary NameNode host上运行`sudo apt-get install hadoop-hdfs-secondarynamenode`

- All cluster hosts except the Resource Manager :`sudo apt-get install hadoop-yarn-nodemanager hadoop-hdfs-datanode hadoop-mapreduce`

- One host in the cluster: `sudo apt-get install hadoop-mapreduce-historyserver hadoop-yarn-proxyserver`

- All client hosts: `sudo apt-get install hadoop-client`


###四，在Cluster中的各个host上安装agent
选择一台主机作为cloudera server服务器，在上面安装：loudera-manager-daemons和cloudera-manager-server：`apt-get install cloudera-manager-daemons cloudera-manager-server`

完成之后启动cloudera manager server的服务：`service cloudera-scm-server start `，能看到提示：`Starting cloudera-scm-server: * cloudera-scm-server started `

如果启动失败可以在`tailf -100 /var/log/cloudera-scm-server/cloudera-scm-server.log`或者`/var/log/cloudera-scm-server/cloudera-scm-server.out`看到日志信息反馈

在cluster的每个host上都需要安装cloudera-manager-agent，用来开启和结束进程，解压配置文件，触发安装。

安装命令：`sudo apt-get install cloudera-manager-agent cloudera-manager-daemons`

在各个host上安装完成之后，需要修改文件`/etc/cloudera-scm-agent/config.ini`来指定cloudera-manager-server的位置，需要修改两个数值：

Property	|Description
----------|-----------
server_host|	Name of the host where Cloudera Manager Server is running.
server_port|	Port on the host where Cloudera Manager Server is running.

在sever host上，是：

```
server_host=localhost
#其他agent host上写server_host的hostname或者ip
server_port=7182
```

其他参数信息详见[官方文档Agent Configuration File](http://www.cloudera.com/documentation/enterprise/latest/topics/cm_ag_agent_config.html#cmug_topic_5_16__section_kw3_5rq_wm)

然后启动cloudera-manager-agent服务：`sudo service cloudera-scm-agent start`

启动失败的各个原因参考[官方文档Troubleshooting Installation and Upgrade Problems](http://www.cloudera.com/documentation/enterprise/latest/topics/cm_ig_troubleshooting.html)

[官网参考](http://www.cloudera.com/documentation/enterprise/latest/topics/cdh_ig_cdh5_install.html#topic_4_4_1_unique_2__p_44_unique_2)

- - -

注：

如果在安装过程中出现了以下错误“ImportError: No module named _io”，不用担心，这是一个已知问题。这是因为CDH5使用的Python版本问题。执行完下面的脚本后，点击重试就可以顺利的完成安装了。如果出现打不开CM Agent的log日志提示，那很可能是你的Host配置有问题，请参考本文最初写的Host配置。

```
mv /usr/lib/cmf/agent/build/env/bin/python /usr/lib/cmf/agent/build/env/bin/python.bak
cp /usr/bin/python2.7 /usr/lib/cmf/agent/build/env/bin/python
```
- - -

#页面安装与配置

1，打开`http://Server-address:7180/`开始安装，用户名密码都是admin，首次启动需要耗时较长

![](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/c1.png)

2，选择Cloudera安装的类型之后，进入添加host集群的页面，输入FQDN
![](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/c2.png)

搜索后之后找到host，确认进入下一步
![](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/c3.png)

3，选择连接方式，在此选了用root+密码的形式登录
![](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/c4.png)

4，确认进入下载过程，耗时较长，错误信息能即时查看排除，一般问题通常找到命令在终端中反复执行即可解决。
![](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/c5.png)

如果想节约时间，可以事先下载好安装的parcels，放入`/opt/cloudera/parcel-repo`路径，可以节约下载时间，parcels的下载地址可以在[http://archive.cloudera.com/cdh5/parcels/](http://archive.cloudera.com/cdh5/parcels/)找到。需要下载的文件有：`CDH-5.6.0-1.cdh5.6.0.p0.45-trusty.parcel.sha，CDH-5.6.0-1.cdh5.6.0.p0.45-trusty.parcel，manifest.json`三个，具体版本号可以自行选择下载。

在安装完成之后可以看到监控画面，至于各个主机角色的分配和一些Health Issue的解决之后补充。
![](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/c6.png)
