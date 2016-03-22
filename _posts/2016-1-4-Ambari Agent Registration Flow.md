---

layout: post
categories: [Apache]
tags: [Big Data, Hadoop, Distributed System, Apache Ambari]

---

Bootstrap：如何在一个host上初始化安装一个Agent和分离注册。其中有两种方法来做Agent上的Bootstrap工作，分别是用SSH和人工手动的非SSH。二者使用的区别在于，用SSH可以在已确认的Host上做Bootstrap，但是需要提供SSH Keys，而人工手动安装则不需要SSH Keys。

Registration：一个Agent host向服务器注册的行为。

- - -

##Bootstrap操作流

**SSH Bootstrap:**

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/16141.png)

**步骤：**

Ambari Server通过调用bootstrap.py来初始化整个bootstrap进程

- Server端通过SSH Keys在Agent上配置Ambari Repo：利用Ambari Server上的*ambari.repo*文件，并且scp到Agent Host上。
- 复制Ambari Agent Setup script：利用scp命令将setupAgent.py脚本复制到Agent host上。
- 在各个Agent上执行Ambari Agent Setup script：SSH到各个Agent Host上然后执行setupAgent.py。
- 在Agent上安装epel-release：用apt-get/yum/zypper工具来安装epel-release包
- 在Agent上安装Ambari-agent：用apt-get/yum/zypper工具来安装Ambari-Agent包
- 配置Ambari-agent.ini：修改`/etc/ambari-agent/conf/ambari-agent.ini`，并设置agent host上的hostname
- 启动Ambari-agent:启动Ambari-agent进程
- 开始Ambari Agent注册：agent开始registration进程

**人工手动Bootstrap**

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/16142.png)

具体步骤内容基本同上

- - -

#Agent注册#

![image](https://raw.githubusercontent.com/kkkelsey/kkkelsey.github.io/master/_images/16143.png)

**步骤**

- 连接握手端口8441：Ambari Agent连接到Ambari Server的握手端口8441。
- 下载Server Certification：Ambari Agent下载Server Certification。
- 请求签署Agent Certification：Ambari Agent请求Ambari Server来签署Agent证书。
- 签署Agent Cert：Ambari Server通过密码签署Agent证书。
- 下载Agent Cert并断掉连接：Ambari Agent下载Agent证书，然后断掉之前的连接。
- 连接注册端口8440：Ambari Agent连接到Ambari Server的注册端口8441
- 用Agent Cert执行2WAY auth：在Agent和Server之间完成2WAY权限认证。
- 获取FQDN：Ambari Agent host获取*Fully Qualified Domain Name（FQDN）*
- 注册Host：利用FQDN，host向Ambari Server提出注册。
- 完成Host注册：Ambari Server完成host的注册过程，把host加入到Ambari数据库
- Agent心跳程序启动：Ambari Agent向Ambari Server开启心跳程序，确认各种命令的执行

- - -
##Reference##

[Using Custom Hostnames](http://docs.hortonworks.com/HDPDocuments/HDP1/HDP-1.2.1/bk_using_Ambari_book/content/ambari-chap7a.html)

[Installing Ambari Agents Manually](http://docs.hortonworks.com/HDPDocuments/HDP1/HDP-1.2.1/bk_using_Ambari_book/content/ambari-chap6.html)