---

layout: post
categories: [C++]
tags: [String,C++,Algorithm,STL]

---


之前一直吵吵着说要刷题刷acm，却一直都在半途而废，我也挺讨厌自己这个样子的，来美国两个月了，感觉自己不能再这样浑浑噩噩地混日子了，Prasad布置的任务也没好好搞，实在有点惭愧的，所以这个礼拜在github上建了一个[Hihocoder](https://github.com/kakack/hihoCoder)的项目，想先从微软的hihocode开始刷，再去做LeetCode的题，在刷的过程中反复准备巩固一些语言和算法上的基础知识，感觉实在是没有时间可以给自己浪费了。

---

我不打算单独将结题的内容当做博客发表，如果您有缘读到这里，可以去我github上自己fork，我提交的代码都是我亲测能够ac的，当然也会有一些是我自己的解题思路，虽然可能不能ac，但是我觉得有保留价值，值得讨论研究。这两天我把Lv.1的简单题都刷过了，基本没涉及到很高深的结构和算法，属于入门型的题目，唯一一个提起我兴趣的是关于字符串处理的#1039题。虽然我在之前也有写过关于字符串的面试博文，但那个更着重在于对字符串基本概念的理解和面试应答，这里我结合了一些优质博文，配上代码，详细罗列和总结一下在C++和Java这两个语言中，关于字符串的应用和方法。

---

## String in C++

从[C++的库函数手册](https://github.com/kakack/kakack.github.io/blob/master/attachment/C%2B%2B%E5%87%BD%E6%95%B0%E6%89%8B%E5%86%8C%2B(LibraryFunctions).chm)上看，String是表示一系列char的一个对象，是basic_string通过char的模板实例化。

在[这里](http://www.cplusplus.com/reference/string/string/)能找到最权威的解释。


### 基本操作符

在C++中， String重载了绝大多数基本操作符，如 +, +=, <, =, , [], <<, >>等。

```C++
String a = "Hello!";

a+="World!"   
a = a + "World!"
//a = "Hello!World!"

if(a < "Hello")
 return true;
//当a字典序排列在“Hello”之前

……

```

String之间可以直接用+相连。而且可以连续+。当系统发现+两段有一个string的时候，会自动把另一个转换成临时的string，执行完之后返回新的string。在对string中某个固定位置的char进行访问的时候，有两种方式，比如`string a="Hello World"`,如果要访问第三个元素，即下标为2的元素，可以用`a[2]`或者`a.at(2)`来访问，前者效率高，后者更稳定。

### find函数

可能是C++ String中用到最多的方法。所以find函数也是有各种各样。

- **find** 查找 
- **rfind** 反向查找 
- **find_first_of** 查找包含子串中的任何字符，返回第一个位置 
- **find_first_not_of** 查找不包含子串中的任何字符，返回第一个位置 
- **find_last_of** 查找包含子串中的任何字符，返回最后一个位置 
- **find_last_not_of** 查找不包含子串中的任何字符

这六个函数还会根据不同的参数传入各被重载四次，所以共有24个find方法。

```C++
size_type find_first_of(const basic_string& s, size_type pos = 0)
size_type find_first_of(const charT* s, size_type pos, size_type n)
size_type find_first_of(const charT* s, size_type pos = 0)
size_type find_first_of(charT c, size_type pos = 0)
```

另外还有一个`substr()`方法，有两个参数，起始位置`int pos`和子串长度`int length`，如果不指定length则默认到结尾。如

```C++
String a = "abcdefg";

cout << a.substr(2, 3);
//"cde"

cout << a.substr(2);
//"cdefg"

```

### string insert, replace, erase

###### Insert:
用于在已有字符串中插入一个新字符串

```C++
string a = "1234567890";

a.insert(2, "Hello");
//12Hello34567890

```

插入的必须是String类型，不能是单个的

###### Replace：
用于替换原有String s中,从int pos开始的int n个内容

`basic _ string& replace( size _ type _Pos1 ,size _ type _Num1 , const value _ type* _Ptr ); `

```C++
string a = "1234567890";

cout << a.replace(2,3,"abc");
//12abc67890

cout << a.replace(2,7,"abc");
//12abc0,如果n的比替换进去的String要长，那么自动填补null进去，占据原有String的位置

string b = "abcdefg";
cout << a.replace(2,2,b,3,3);
//12def67, 也可以指定替代进去的String的子串位置

```

###### Erase：
用于删除字符

```C++
string a = "abcdef";
s.erase(3);
//abc, 从下标3到之后全部删除

string a = "abcdef";
s.erase(3,2);
//abcf, 从下标3后删除2个字符

```
- - -

### 附录

```C++
string 函数列表 函数名 描述
begin 得到指向字符串开头的Iterator
end 得到指向字符串结尾的Iterator
rbegin 得到指向反向字符串开头的Iterator
rend 得到指向反向字符串结尾的Iterator
size 得到字符串的大小
length 和size函数功能相同
max_size 字符串可能的最大大小
capacity 在不重新分配内存的情况下，字符串可能的大小
empty 判断是否为空
operator[] 取第几个元素，相当于数组
c_str 取得C风格的const char* 字符串
data 取得字符串内容地址
operator= 赋值操作符
reserve 预留空间
swap 交换函数
insert 插入字符
append 追加字符
push_back 追加字符
operator+= += 操作符
erase 删除字符串
clear 清空字符容器中所有内容
resize 重新分配空间
assign 和赋值操作符一样
replace 替代
copy 字符串到空间
find 查找
rfind 反向查找
find_first_of 查找包含子串中的任何字符，返回第一个位置
find_first_not_of 查找不包含子串中的任何字符，返回第一个位置
find_last_of 查找包含子串中的任何字符，返回最后一个位置
find_last_not_of 查找不包含子串中的任何字符，返回最后一个位置
substr 得到字串
compare 比较字符串
operator+ 字符串链接
operator== 判断是否相等
operator!= 判断是否不等于
operator< 判断是否小于
operator>> 从输入流中读入字符串
operator<< 字符串写入输出流
getline 从输入流中读入一行
```

