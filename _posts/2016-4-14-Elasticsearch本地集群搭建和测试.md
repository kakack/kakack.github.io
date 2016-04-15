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


然后把`target/releases/elasticsearch-analysis-ik-{version}.zip`解压到`your-es-root/plugins/ik`，目标文件夹需要自己手动创建。比如在我本地用的路劲就是`/home/ambari/elasticsearch-2.3.1/plugins/ik`

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

- - -
#请求JSON

其他的语言可以通过9200端口与 Elasticsearch 的 RESTful API 进行通信。事实上，如你所见，你甚至可以使用行命令 curl 来与 Elasticsearch 通信。

```
      <1>     <2>                   <3>    <4>
curl -XGET 'http://localhost:9200/_count?pretty' -d '
{  <5>
    "query": {
        "match_all": {}
    }
}
'

```

- <1> 相应的 HTTP 请求方法 或者 变量 : GET, POST, PUT, HEAD 或者 DELETE。
- <2> 集群中任意一个节点的访问协议、主机名以及端口。
- <3> 请求的路径。
- <4> 任意一个查询后再加上 ?pretty 就可以生成 更加美观 的JSON反馈，以增强可读性。
- <5> 一个 JSON 编码的请求主体（如果需要的话


####Example

创建一个index：

```
curl -XPUT http://localhost:9200/index?pretty

```

创建mapping：

```
curl -XPOST http://localhost:9200/index/fulltext/_mapping -d'
{
    "fulltext": {
             "_all": {
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_max_word",
            "term_vector": "no",
            "store": "false"
        },
        "properties": {
            "content": {
                "type": "string",
                "store": "no",
                "term_vector": "with_positions_offsets",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_max_word",
                "include_in_all": "true",
                "boost": 8
            }
        }
    }
}'

``` 

插入数据：

```
curl -XPOST http://localhost:9200/index/fulltext/1 -d'
{"content":"美国留给伊拉克的是个烂摊子吗"}
'

curl -XPOST http://localhost:9200/index/fulltext/2 -d'
{"content":"公安部：各地校车将享最高路权"}
'

curl -XPOST http://localhost:9200/index/fulltext/3 -d'
{"content":"中韩渔警冲突调查：韩警平均每天扣1艘中国渔船"}
'

curl -XPOST http://localhost:9200/index/fulltext/4 -d'
{"content":"中国驻洛杉矶领事馆遭亚裔男子枪击 嫌犯已自首"}
'

```

查询：

```
curl -XPOST http://localhost:9200/index/fulltext/_search?pretty  -d'
{
    "query" : { "term" : { "content" : "中国" }},
    "highlight" : {
        "pre_tags" : ["<tag1>", "<tag2>"],
        "post_tags" : ["</tag1>", "</tag2>"],
        "fields" : {
            "content" : {}
        }
    }
}
'

```

返回结果：

```
{
    "took": 14,
    "timed_out": false,
    "_shards": {
        "total": 5,
        "successful": 5,
        "failed": 0
    },
    "hits": {
        "total": 2,
        "max_score": 2,
        "hits": [
            {
                "_index": "index",
                "_type": "fulltext",
                "_id": "4",
                "_score": 2,
                "_source": {
                    "content": "中国驻洛杉矶领事馆遭亚裔男子枪击 嫌犯已自首"
                },
                "highlight": {
                    "content": [
                        "<tag1>中国</tag1>驻洛杉矶领事馆遭亚裔男子枪击 嫌犯已自首 "
                    ]
                }
            },
            {
                "_index": "index",
                "_type": "fulltext",
                "_id": "3",
                "_score": 2,
                "_source": {
                    "content": "中韩渔警冲突调查：韩警平均每天扣1艘中国渔船"
                },
                "highlight": {
                    "content": [
                        "均每天扣1艘<tag1>中国</tag1>渔船 "
                    ]
                }
            }
        ]
    }
}

```