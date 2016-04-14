---

layout: post
categories: [ElasticSearch]
tags: [Linux, ElasticSearch]

---

#Elastic Search

建立在全文搜索引擎[Apache Lucene(TM)](https://lucene.apache.org/core/)基础上，被誉为当今最先进高效的全功能开源搜索引擎框架。

- 分布式实时文件存储，并将每一个字段都编入索引，使其可以被搜索。
- 实时分析的分布式搜索引擎。
- 可以扩展到上百台服务器，处理PB级别的结构化或非结构化数据。


#安装Elastic Search

前置安装：jdk1.7+，maven；

*需要注意的是，ES不允许`root`用户启动服务，所以在做以下操作时

进入[下载页面](https://www.elastic.co/downloads/elasticsearch)下载当前最高版本的ES，以成文时的版本2.3.1为例：

```
$ wget https://download.elastic.co/elasticsearch/release/org/elasticsearch/distribution/zip/elasticsearch/2.3.1/elasticsearch-2.3.1.zip
$ unzip elasticsearch-2.3.1.zip
$ cd elasticsearch-2.3.1 
$ ./bin/elasticsearch


```

---

#安装IK Analysis for ElasticSearch

IK Analysis是ES的一个插件，用来支持用户自定义的dictionary。

可直接从git上clone下来后本地编译

```
$ git clone https://github.com/medcl/elasticsearch-analysis-ik.git
$ cd elasticsearch-analysis-ik
$ mvn clean
$ mvn compile
$ mvn package

```


然后把`target/releases/elasticsearch-analysis-ik-{version}.zip`解压到`your-es-root/plugins/ik`，目标文件夹需要自己手动创建。

最后手动重启elasticsearch服务

---
#测试

创建一个index

```
$ curl -XPUT http://localhost:9200/index
```

能够得到以下反馈，表示所有服务都正常运行，可以进行之后的实验了。

```
{
   "status": 200,
   "name": "Shrunken Bones",
   "version": {
      "number": "1.4.0",
      "lucene_version": "4.10"
   },
   "tagline": "You Know, for Search"
}
```