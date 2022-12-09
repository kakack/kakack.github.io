---

layout: post
categories: [dairy]
tags: [Dairy]

---

## 1.	Abstract

Profiing的本质目的是方便用户能够准确掌握程序耗时和函数之间的调用结构，便于了解程序运行性能瓶颈进而后续优化。一个profiler常在程序运行的过程中收集运行信息，通常分为这些步骤：

1. 以profiling enabled的模式编译和运行目标程序；
2. 执行程序生成profile data file；
3. 运行gprof来分析profile data。



## 2.	Principle & Detail

### 2.1	Implement

Profiler通常改变程序中每一个方法的编译方式，从而在每一个方法被调用的时候，会stash一些调用信息，从而方便profiler找到该方法的调用者，并统计调用次数。当编译时加了`-pg`选项之后，每个方法都会第一个调用操作`mcount`（或成为`_mcount`或`__mcount`），这个操作记录了一个in-memory call graph table中该方法的父辈和祖辈方法。这通常通过检查堆栈帧stack frame中找到子地址和返回的父地址，`mcount`本身是个简单的汇编语言stub rountine，通过参数`frompc`和`selfpc`调用`__mcount_internal`，负责维护内存中的调用图。

### 2.2	Profiling Data File Format

新的文件格式在`gmon_out.h`中定义，由三部分组成：

#### 2.2.1	Histogram Records

Histogram Records由一个header和一个bins的array组成，其中header包含直方图跨越的文本范围、直方图大小、分析时钟的速率和物理尺寸。直方图的bin是16-bit的数字，每个bin代表等量的文本空间。例如，如果文本段的长度为 1000 个字节，并且直方图中有 10 个 bin，则每个 bin 代表 100 个字节。

#### 2.2.2	Call-Graph Records

Call-Graph Records的格式与BSD-derived文件格式中使用的格式相同。它由call graph中的arc和程序执行期间这个arc被遍历次数的计数组成。一个arc由一对地址组成：第一个必须在调用者的函数内，第二个必须在被调用者的函数内。在函数级别执行分析时，这些地址可以指向相应函数内的任何位置。 但是，在行级别进行分析时，地址最好尽可能靠近调用站点/入口点。 这将确保行级调用图能够准确识别哪一行源代码执行了对函数的调用。

#### 2.2.3	Basic-Block Excution Count Records

Basic-Block Excution Count Records由一个header和后续的一系列地址/计数对组成。这个header仅指定序列的长度。在地址/计数对中，地址标识基本块，计数指定基本块执行的次数。 可以使用基本地址中的任何地址。

### 2.3	Details

`gprof`是首先会处理那些option，在这个阶段如果option中指定了使用哪个symspecs，那么它会构建自己的symspec list（sym_ids.c：sym_id_add）。`gprof`维护了一个symspecs的单向链表，最后会得到12个symbol table，组织成6个include/exclude pair，每一个pair对应：

	-	flat profile(INCL_FLAT/EXCL_FLAT)
	-	call graph arcs (INCL_ARCS/EXCL_ARCS)
	-	printing in the call graph (INCL_GRAPH/EXCL_GRAPH)
	-	timing propagation in the call graph (INCL_TIME/EXCL_TIME)
	-	the annotated source listing (INCL_ANNO/EXCL_ANNO)
	-	the execution count listing (INCL_EXEC/EXCL_EXEC).

在option处理之后，gprof 通过将 default_excluded_list 中的所有symspecs添加exclude lists EXCL_TIME 和 EXCL_GRAPH 来完成构建symspec list。

之后调用BFD library来打开目标文件，并验证，读取它的symbol table(core.c:core_init)，在申请到一个合适大小的symbol array后使用`bfd_canonincalize_symtab`。在这时，function mappings被读取，core text space被读取到内存。

此时gprof自己的symbol table（一个Sym structure array）被构建完成。

## 3.	Output Content

以gprof为例：

### 3.1	Flat profile

| 属性标签           | 说明                                                       |
| ------------------ | ---------------------------------------------------------- |
| % time             | 函数使用时间占所有时间的百分比。                           |
| cumulative seconds | 函数和上列函数累计执行的时间。                             |
| self seconds       | 函数本身所执行的时间。（每次调用的累计）                   |
| calls              | 函数被调用的次数                                           |
| self ms/call       | 每一次调用花费在函数的时间microseconds。                   |
| total ms/call      | 每一次调用，花费在函数及其衍生函数的平均时间microseconds。 |
| name               | 函数名。                                                   |



### 3.1	Call Graph

| 属性标签 | 说明                                       |
| -------- | ------------------------------------------ |
| index    | 索引值                                     |
| %time    | 函数消耗的时间占所有时间的百分比           |
| self     | 函数本身花费的时间（不包括调用的衍生函数） |
| children | 调用的衍生函数所花费的时间                 |
| called   | 调用次数                                   |
| name     | 函数名。                                   |



## 4	GProf Example

### 4.1	Example code

```c
//test_gprof_new.c
#include<stdio.h>

void new_func1(void) {
    printf("\n Inside new_func1()\n");
    int i = 0;
    for(;i<0xffffffee;i++);
    return;
}
```



```c
//test_gprof.c
#include<stdio.h>

void new_func1(void);

void func1(void) {
    printf("\n Inside func1 \n");
    int i = 0;
    for(;i<0xffffffff;i++);
    new_func1();
    return;
}

static void func2(void) {
    printf("\n Inside func2 \n");
    int i = 0;
    for(;i<0xffffffaa;i++);
    return;
}

int main(void) {
    printf("\n Inside main()\n");
    int i = 0;
    for(;i<0xffffff;i++);
    func1();
    func2();
    return 0;
}
```



### 4.2	Process

```
$ gcc -Wall -pg test_gprof.c test_gprof_new.c -o test_gprof
# 编译目标代码
$ ls
test_gprof  test_gprof.c  test_gprof_new.c

$ ./test_gprof 

 Inside main()

 Inside func1 

 Inside new_func1()

 Inside func2 

# 执行编译出的可执行文件

$ ls
gmon.out  test_gprof  test_gprof.c  test_gprof_new.c

$  gprof test_gprof gmon.out > analysis.txt
$ ls
analysis.txt  gmon.out  test_gprof  test_gprof.c  test_gprof_new.c
# 利用gprof将执行结果输出整理成可阅读的txt文件
```



### 4.3	Output File

输出文件即`analysis.txt`，其内容可以分成两部分：

1. Flat profile
2. Call graph

关于具体content可以使用如`-Q`、`-q`、`-p`、`-P`、`-b`、`-a`等标签优化或省略规整输出内容，暂不赘述。

#### 4.3.1	Flat profile

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20221205_5.jpg)

#### 4.3.2	Call graph

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20221205_6.jpg)



## 5 Other Profiling Tools

- valgrind 
- perf



## 6 Summary



对于单个function输出profiling统计信息：

 - 耗时time
   - 该function耗时绝对时间
   - 该function耗时占比%；
   - 该function和上列函数累计耗时总和。
 - 调用calls
   - 该function被调用次数；
   - 每次调用在该function上耗时
   - 每次调用该function及其衍生函数平均耗时



对于前端呈现chart功能：

- 全量function概览，呈现function name、调用与返回时间点、执行耗时；
- 单个function详情列表，含单个function具体统计信息；
- 上下文选择；
- 查找function；
- 其他点击选中优化显示等





