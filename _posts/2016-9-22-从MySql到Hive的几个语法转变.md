---

layout: post
categories: [Hadoop]
tags: [Hadoop, Hive]

---

这几天在做`sqlworkflow`的过程中，接手了把之前只支持MySql的Workflow扩展为支持Hive和Oracle数据库，后者我没接触过，所以从跟MySql语法极其相似的Hive着手。

---

#Hive不支持行级别的insert

这个是使用过程中就最早能感受到的一个区别，hive中插入数据以`load`操作为主，如

```hive
LOAD DATA [LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename \
[PARTITION (partcol1=val1, partcol2=val2 ...)]
```

- 其中`filepath`可以是：
	- 相对路径，如`project/data1`
	- 绝对路径，如`/tmp/cache/data1`
	- 一个完整的url，如`hdfs://namenode:9000/user/hive/project/data1`
- 被导入目标可以是一个表或者分区，如果表是有分区的，那必须指定好表所在的特定分区
- `filepath`指定的可以是一个文件或者一个目录
- 如果用了`local`，那么会从本地文件夹系统寻址，会复制`filename`指向的所有的文件到目标文件系统
- 如果用了`overwrite`，那么如果原先有同名的表，则会被删掉重新写一遍。

Hive也支持`insert`语句通过查询过程插入数据：

```hive
INSERT OVERWRITE TABLE tablename1 \
[PARTITION (partcol1=val1, partcol2=val2 ...) [IF NOT EXISTS]] \
select_statement1 FROM from_statement;
```

同样Hive也不支持行级别的update和delete操作

---

#Hive对分号异常敏感

虽然MySql和Hive都是用分号结束一句query，但是hive对于分号的敏感程度远超过mysql，缺乏智能识别，所以当语句中出现分号时，需用`\073`代替

---

#IS [NOT] NULL

SQL中null代表空值, 值得警惕的是, 在HiveQL中String类型的字段若是空(empty)字符串, 即长度为0, 那么对它进行IS NULL的判断结果是False。

---

#聚合函数

语句通常跟聚合函数一起使用，按照一个或者多个列对结果进行分组，然后对每个组执行聚合操作。然而在hive中不允许访问非group by的columns，所以如果要在一个有group by的query里查询非group by的属性，会用`collect_set(属性名)[0]`的方法，例如：

```hive
select collect_set(id)[0],collect_set(name)[0],salary,avg(salary) \
from table employee group by salary;
```
---
#IN

SQL中可以使用IN操作符来规定多个值：

```
SELECT * FROM Persons WHERE LastName IN ('Adams','Carter');
```
HiveQL目前是不支持IN操作符的，需要通过转换为多个OR连接的条件：

```
SELECT * FROM Persons WHERE LastName = 'Adams' OR LastName = 'Carter';
```

---

#INNER JOIN

在之前版本的Hive中，下面这种写法是不对的，是只在mysql中支持的：

```
select a.name, a.salary, b.edu from employee a, person b where a.name=b.name;
```

需要写成：

```
select a.name, a.salary, b.edu from employee a join person b on a.name=b.name;
```

但是在新版本的Hive中已经支持了第一种写法。