---

layout: post
categories: [workflow]
tags: [Linux, workflow]

---

Workflow Engine 设计文档
==================

背景
----

在大数据的背景下, 我们部署了hadoop生态圈中一系列的组件, 但离真正的用户可用还有
很长的距离, 这其中就包含了如何让不懂编程的用户可以方便的进行工作流的管理和执行.

所谓的工作流(flow), 是指将一些常用的任务(job)作为一个个节点, 按照一定的执行顺序组织起来的
工作流程. 在执行这个工作流程的时候, 会按照设计的步骤, 依次执行.

使用工作流的好处在于: 

* 一次设计, 多次重用
* 可以为一个工作流设置运行周期(每天1点执行...), 定时执行

需求
-----

针对数据分析师这类人群, 他们熟悉的是各种机器学习算法而非算法的实现细节, 
所以他们需要的是一种能够方便他们调参的工具, 不必关心代码的细节. 具体的需求
包含如下: 

* 调参
* 比对各种模型的学习效果
* 可视化得展示结果

针对一些数据管理人员, 存在这样的需求:

* 每天新增的数据要定时导入到数据仓库中
* 新增的数据会需要经过相同的处理后才能导入仓库中
* 需要一些可视化的报表来追踪当前仓库的状态

开发目标
--------

针对需求，我们的目标是实现一个基于模版的DAG工作流调度引擎WorkFlow, 工作流引擎需要包含如下的功能:

### 通用的任务模板

* ETL任务
* 数据处理类任务
* 机器学习类任务
* 报表类任务

### 用户友好的流程设计界面

* 工作流的保存, 执行, 更新, 重用, 查看执行历史功能
* 任务的参数配置, 部分参数的自动匹配
* 基本的验证功能
    * 是否有向无环
    * 参数是否都配置好
    * 任务之间是否有有歧义的地方
* 工作流执行状态的实时反馈, 日志信息的显示

### 透明化的中间结果集的处理

* 对于一连串的数据处理，用户可能只关心开始的输入和最终的结果，而中间的结果集就可以透明化，用户无需关心它们的具体位置



技术路线
--------

![workflow server 框架图](resources/workflow-architecture.png)


azkaban 是:
* LinkedIn开源的批量工作流任务调度器
* 可设置工作流的运行顺序和调度策略
* KV文件格式来建立任务之间的依赖关系

特点：
* 友好的Web界面
* 用户认证和权限验证
* 能够杀死并重启工作流
* 模块化和可插拔的插件机制
* 日志记录
* 任务执行结果邮件报警

yellowbook server 是:
* 通用的数据字典，资源的保存和查询（TODO）

Workflow Server 通过提供一系列的 RESTful API 来供客户端调用, 包括:

* flow 的增删查改
* 将flow打包成zip并上传azkaban执行
* job 的增删查改

借助 azkaban 来: 

* 执行工作流
* 查看flow执行状态和结果
* 调度工作流(定时/周期性任务)

借助 yellowbook server 提供的服务来:

* 查询资源摘要
* 查询资源详细信息
* 增删查改特定资源(flow, job, schema, file...)


设计思路
--------

### 总体设计

### 任务模板(Task)的设计

以`hive源`模板为例, 此模板的作用是从hive数据库中选取已经存在的表作为输出,
供后面的任务使用.

```json
{
  "namespace": "DataSource",
  "localName": "selectTable",
  "alias": "hive源",
  "description": "从大数据平台中选择一个表作为数据源",
  "type": "Task",
  "template": "command = echo Select table @tableOut",
  "category": "DATASOURCE",
  "paramList": {
      "tableOut": {
          "name": "hive表",
          "javaType": "Resource"
      }
  },
  "inputs":{},
  "outputs": {
    "tableOut": {
      "alias":"表1",
      "type": "Schema",
      "detailed":false
    }
  },
  "ioResMappings": {}
}
```

下面逐字段的进行分析:

* namespace 将模板按照所属的类别划分, 比如说分成: DataSource, DataSource, ETL等.
* localName 是模板在namespace类别下, 具体的英文名称
* namespace + localName 在整个模板空间中保证唯一
* alias 字段是模板的中文名, 供前端显示用
* description 用来存放模板的介绍信息
* type 字段值为 "Task", 是模板对应的java类型
* template 是模板对应的命令实体, 其中以@开头的单词是命令中空出的变量
* category 字段是模板对应的大类, 目前分为: DATASOUCE, TRANSFORMER, DATATARGET, 
分别用来表示数据源类, 数据处理类, 数据消耗类.
* paramList 是一个参数类表, 前面template字段中以@开头的单词会在这里详细说明参数的
相关信息, 包括: 数据类型, 前端显示的类型, 默认值, 可选值等
* inputs 和 outputs 字段用来申明输入或者输出的资源, 可以有一个或多个, 
目前支持的资源类型包括文件, 数据库表.
* ioResMappings 是一个映射表, 专门用来指定输出资源与输入资源之间的特定关系, 
比如: 输出资源中的某个表与输入资源中的一个表有相同的格式, 就可以在这里面指明.

```json
{
  "namespace": "DataProcess",
  "localName": "sampling",
  "alias": "随机采样",
  "description": "随机采样",
  "type": "Task",
  "template": "command=/hadoop/hive/bin/hive -e 'insert into table @tableOut select * from @tableIn where rand() < @percentage'",
  "category": "TRANSFORMER",
  "paramList": {
  	"percentage" : {
  		"name" : "采样比率",
  		"javaType": "double",
        "defaultValue":"0.1"
  	}
  },
  "inputs": {
    "tableIn": {
      "alias":"表1",
      "type": "Schema",
      "detailed":false
    }
  },
  "outputs": {
  	"tableOut" : {
        "alias":"表2",
  		"type":"Schema",
      	"detailed":false
  	}
  },
  "ioResMappings": {
  	"tableOut": "tableIn"
  }
}
```

现在详细介绍一下`随机采样`这个模板, 由namespace可知这个模板属于"DataProcess"组, 
且它是这个组中的"sampling"模板, 中文名为"随机采样", 属于程序中的 Task 类型,
在现有的分类中, 属于 "TRANSFORMER" 类型, 也就是数据转换类模板. 
template字段中显示了需要执行的操作, 这个一条hive sql命令, 从 tableIn 表中选取
percentage 这么多比率的数据量, 放入 tableOut 表中, 这三个参数的具体信息分别在: 
paramList, inputs, outputs中详细指明. 

可以看到, 在 paramList 中给出了参数 percentage 的详细信息, 它对应的 java
类型为 double, 默认值为 0.1, 供前端显示的中文名为"采样比率". 在 inputs 中详细介绍了 tableIn
表的信息, 首先在 type 类型中可以看到这是一个数据库表, detailed 字段告诉我们这张表并没有指定具体的结构,
具体的结构需要查看它的前一个任务的输出才能知道. outputs 中的信息和 inputs 中的类似. 
最后可以看到在 ioResMappings 中指定了 tableOut 和 tableIn 的对应关系, 据此可以知道 tableOut
的结构和 tableIn 是相同的.


### 工作流(flow)的设计

下面介绍工作流, 一个 flow 有下面几部分组成: 

```json
{
    "name": "",
    "user":"",
    "jobs":{},
    "origins":[],
    "mappings":{}
}

```

* name 是这个 flow 的名称
* user 是创建这个 flow 的用户
* jobs 里面包含了所有的任务, 这些任务和他们的名字组成了键值对
* origins 里面存放流程的所有起始点
* mappings 用来存放用户指定的资源之间的对应关系

下面具体介绍 flow 中任务的描述部分: 

```json

"sampling": {
    "position":{
        "x":20,
        "y":20
    },
    "children": [
        "selectAllFromTable2"
    ],
    "content": "",
    "name": "sampling",
    "outputs": {
        "tableOut": {
            "detailed": "false",
            "type": "Schema"
        }
    },
    "params": {
        "percentage": "0.1"
    },
    "parents": [
        "selectTable"
    ],
    "task": "DataProcess/sampling"
}

```

如前所述, flow 中的 jobs 字段由任务的名称和任务的详细描述构成的键值对组成,
任务的描述部分包括了一下这些项目: 

* task 字段指明了这个任务对应的是模板系统中的哪个模板
* name 部分是这个任务的名称, 实际应用中会使用 UUID 来填写
* parents 中填写了这个任务的所有父节点
* children 中列出了和自己直接相连的子任务名称
* params 中包含了用户填写的参数值
* outputs 中指出了这个任务要生成资源的相关信息
* position 是任务在画布上的位置信息, 由x和y决定其在二维画布上的位置
* content 里面存放最终要执行的完整命令, 此处都初始化为长度为0的字符串


一个可以执行的 flow 的json描述如下, 这个 flow 的拓扑结构如下图所示:

![flow 的拓扑结构](resources/classic-flow-example.png)

```json
{
    "jobs": {
        "newFLowExample": {
            "position":{
                "x":10,
                "y":10
            }, "children": [], "content": "",
            "name": "newFlowExample",
            "outputs": {},
            "params": {
                "flowName": "newFlowExample"
            },
            "parents": [
                "selectAllFromTable",
                "selectAllFromTable2"
            ],
            "task": "Hide/FlowEnd"
        },
        "sampling": {
            "position":{
                "x":20,
                "y":20
            },
            "children": [
                "selectAllFromTable2"
            ],
            "content": "",
            "name": "sampling",
            "outputs": {
                "tableOut": {
                    "detailed": "false",
                    "type": "Schema"
                }
            },
            "params": {
                "percentage": "0.1"
            },
            "parents": [
                "selectTable"
            ],
            "task": "DataProcess/sampling"
        },
        "selectAllFromTable": {
            "position":{
                "x":30,
                "y":30
            },
            "children": [
                "newFLowExample"
            ],
            "content": "",
            "name": "selectAllFromTable",
            "outputs": {},
            "params": {},
            "parents": [
                "selectTable"
            ],
            "task": "DataProcess/selectAll"
        },
        "selectAllFromTable2": {
            "position":{
                "x":40,
                "y":40
            },
            "children": [
                "newFLowExample"
            ],
            "content": "",
            "name": "selectAllFromTable2",
            "outputs": {},
            "params": {},
            "parents": [
                "sampling"
            ],
            "task": "DataProcess/selectAll"
        },
        "selectTable": {
            "position":{
                "x":50,
                "y":50
            },
            "children": [
                "selectAllFromTable",
                "sampling"
            ],
            "content": "",
            "name": "selectTable",
            "outputs": {
                "tableOut": {
                    "detailed": "true",
                    "meta": {
                        "fields": [
                            {
                                "comment": "frequency of this word",
                                "name": "freq",
                                "type": "int"
                            },
                            {
                                "comment": "english word",
                                "name": "word",
                                "type": "string"
                            }
                        ]
                    },
                    "type": "Schema"
                }
            },
            "params": {
                "tableOut": "words"
            },
            "parents": [],
            "task": "DataSource/selectTable"
        }
    },
    "mappings": {},
    "name": "newFlowExample",
    "origins": [
        "selectTable"
    ],
    "user": "admin"
}
```

