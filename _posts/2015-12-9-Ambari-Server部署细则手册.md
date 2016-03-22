---

layout: post
categories: [Apache]
tags: [Big Data, Hadoop, Distributed System, Apache Ambari]

---

#Ambari-Server部署细则手册
- - -

预期平台：Ubuntu 14（全新）

预期集群：暂时单点

示例ip：10.10.245.31

- - -

##前置安装

- 开发环境：

	1. Python: 建议在Python 2.7.6以上版本，因为我在2.7.3上出现过失败，在ambari server之后某个组件安装过程中出现依赖问题。因为Ubuntu 14预装了Python2.7.6，为保险起见，可：`apt-get install python-dev`以确定Python环境。
	2. Java 7：一般建议用OracleJDK比较方便

```
java -version
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java7-installer
sudo apt-get install oracle-java7-set-default

```

- 数据库：MySQL：`sudo apt-get install mysql-server`

```
create database ambari;
use ambari;
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'youpassword' WITH GRANT OPTION;
FLUSH PRIVILEGES;
exit;
```

修改/etc/mysql/my.conf

将skip-external-locking注释掉，并将bind-address = 0.0.0.0

重启MySQL 服务`/etc/init.d/mysql restart`

- 设置hostname: `hostname ambari31`，然后在/etc/hosts中，将原有的localhost和ubuntu删去，添加一行ip+hostname+FQDN，例如`192.168.***.** ambari31 ambaritest ambari.test.com`，再添一行`127.0.0.1 localhost`然后修改/etc/hostname，将原有的hostname替换为想要的

- 安装ntp：`sudo apt-get install ntp`，重启服务'sudo service ntp restart'

- 可考虑安装selinux：`sudo apt-get install selinux`，在ambari开始装之前会被检测但是亲测似乎并无别的用处




- - -

##安装Apache Ambari
对于Ubuntu 14用户来说，首先进入Ambari repository下载

`cd /etc/apt/sources.list.d`

`wget http://public-repo-1.hortonworks.com/ambari/ubuntu14/2.x/updates/2.1.2/ambari.list`

接着使用apt-get命令安装

```
  apt-key adv --recv-keys --keyserver keyserver.ubuntu.com B9733A7A07513CAD
  apt-get update
  apt-get install ambari-server
```

安装完毕之后，将mysql-jdbc-connector放到/usr/share/java目录下

关于jdbc这点，一开始我一直以为是我放的版本不对，所以花了大量时间在尝试不同版本的jar，后来找到一篇教程，说Ambari server已经给你准备好了需要的jdbc，只要将其复制到`/usr/lib/ambari-server`下

`cp /var/lib/ambari-server/resources/ojdbc6.jar /usr/lib/ambari-server`

否则日志会报错：

```
 29 Sep 2015 10:51:29,225 ERROR main DBAccessorImpl:102-
Error while creating database accessor
java.lang.ClassNotFoundException:oracle.jdbc.driver.OracleDriver

```


- - -
##部署和启动

用户设置：`ambari-server setup`

一直按照提示进行，jdk建议选择jdk7，数据库建议使用MySQL，然后在mysql中创建数据库ambari，在该数据库中导入sql ddl：

`source /var/lib/ambari-server/resources/Ambari-DDL-MySQL-CREATE.sql`

期间需要设置一些数据库名（建议用之前创建的ambari），以及数据库用户名和密码等，最后在结束时会提示successful。

启动服务：`ambari-server start`

自动启动，如果jdbc报错那么将新的mysql-connector和jdbc复制到/usr/share/java路径下，mysql-connector建议使用版本较高的5.1.37，官网都能下到。

如果失败报错可通过`cat /var/log/ambari-server/ambari-server.log`查看

一般错误都出在jdbc和mysql上

- - -

##启动后页面配置

通过浏览器访问`http://ip:8080`来访问，初始的账号密码都是admin

之后依照步骤依次执行，添加ssh-key的地方注意需要私钥文件，其余没什么要注意的

在选择组件hive时，可先用root账号在mysql中建立好名为hive的数据库

最后选择组件安装，安装过程处诸有错误，可按照提示修复。

