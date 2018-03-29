---

layout: post
categories: [Python]
tags: [Python, RegEx]

---


正则表达式是一个特殊的字符序列，它能帮助你方便的检查一个字符串是否与某种模式匹配。

Python 自1.5版本起增加了re 模块，它提供 Perl 风格的正则表达式模式。

re 模块使 Python 语言拥有全部的正则表达式功能。

compile 函数根据一个模式字符串和可选的标志参数生成一个正则表达式对象。该对象拥有一系列方法用于正则表达式匹配和替换。

re 模块也提供了与这些方法功能完全一致的函数，这些函数使用一个模式字符串做为它们的第一个参数。

本章节主要介绍Python中常用的正则表达式处理函数。
 
 - - -
 
 
# re.match和re.search函数

- re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。
- `re.match(pattern, string, flags=0)`

参数	|	描述
---|---
pattern	|	匹配的正则表达式
string	|	要匹配的字符串。
flags	|	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。
pos| 文本中正则表达式开始搜索的索引。值与Pattern.match()和Pattern.seach()方法的同名参数相同。
endpos| 文本中正则表达式结束搜索的索引。值与Pattern.match()和Pattern.seach()方法的同名参数相同。
lastindex| 最后一个被捕获的分组在文本中的索引。如果没有被捕获的分组，将为None。
lastgroup| 最后一个被捕获的分组的别名。如果这个分组没有别名或者没有被捕获的分组，将为None。

方法：

- **group([group1, …]):**
获得一个或多个分组截获的字符串；指定多个参数时将以元组形式返回。group1可以使用编号也可以使用别名；编号0代表整个匹配的子串；不填写参数时，返回group(0)；没有截获字符串的组返回None；截获了多次的组返回最后一次截获的子串。
- **groups([default]):**
以元组形式返回全部分组截获的字符串。相当于调用group(1,2,…last)。default表示没有截获字符串的组以这个值替代，默认为None。
- **groupdict([default]):**
返回以有别名的组的别名为键、以该组截获的子串为值的字典，没有别名的组不包含在内。default含义同上。
- **start([group]):** 
返回指定的组截获的子串在string中的起始索引（子串第一个字符的索引）。group默认值为0。
- **end([group]):**
返回指定的组截获的子串在string中的结束索引（子串最后一个字符的索引+1）。group默认值为0。
- **span([group]):** 
返回(start(group), end(group))。
- **expand(template):** 
将匹配到的分组代入template中然后返回。template中可以使用\id或\g<id>、\g<name>引用分组，但不能使用编号0。\id与\g<id>是等价的；但\10将被认为是第10个分组，如果你想表达\1之后是字符'0'，只能使用\g<1>0。

```Python
import re
m = re.match(r'(\w+) (\w+)(?P<sign>.*)', 'hello world!')
 
print("m.string:", m.string)
print("m.re:", m.re)
print("m.pos:", m.pos)
print("m.endpos:", m.endpos)
print("m.lastindex:", m.lastindex)
print("m.lastgroup:", m.lastgroup)
 
print("m.group(1,2):", m.group(1, 2))
print("m.groups():", m.groups())
print("m.groupdict():", m.groupdict())
print("m.start(2):", m.start(2))
print("m.end(2):", m.end(2))
print("m.span(2):", m.span(2))
print(r"m.expand(r'\2 \1\3'):", m.expand(r'\2 \1\3'))
 
### output ###
# m.string: hello world!
# m.re: <_sre.SRE_Pattern object at 0x016E1A38>
# m.pos: 0
# m.endpos: 12
# m.lastindex: 3
# m.lastgroup: sign
# m.group(1,2): ('hello', 'world')
# m.groups(): ('hello', 'world', '!')
# m.groupdict(): {'sign': '!'}
# m.start(2): 6
# m.end(2): 11
# m.span(2): (6, 11)
# m.expand(r'\2 \1\3'): world hello!
```
 
- - -
 
- re.search 扫描整个字符串并返回第一个成功的匹配。
- `re.search(pattern, string, flags=0)`

参数	|	描述
---|---
pattern	|	匹配的正则表达式
string	|	要匹配的字符串。
flags	|	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。
groups	| 表达式中分组的数量。
groupindex	| 以表达式中有别名的组的别名为键、以该组对应的编号为值的字典，没有别名的组不包含在内。

```Python
p = re.compile(r'(\w+) (\w+)(?P<sign>.*)', re.DOTALL)
 
print("p.pattern:", p.pattern)
print("p.flags:", p.flags)
print("p.groups:", p.groups)
print("p.groupindex:", p.groupindex)
 
### output ###
# p.pattern: (\w+) (\w+)(?P<sign>.*)
# p.flags: 16
# p.groups: 3
# p.groupindex: {'sign': 3}
```

re.match与re.search的区别：re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配。

- - -

# 范围匹配

如果需要找到潜在的多个可能性文字, 我们可以使用 `[]` 将可能的字符囊括进来. 比如 `[ab]` 就说明我想要找的字符可以是 `a` 也可以是 `b`. 这里我们还需要注意的是, 建立一个正则的规则, 我们在 `pattern` 的 `“”` 前面需要加上一个 `r` 用来表示这是正则表达式, 而不是普通字符串. 通过下面这种形式, 如果字符串中出现 `“run”` 或者是 `“ran”`, 它都能找到.

```Python
# multiple patterns ("run" or "ran")
ptn = r"r[au]n"       
# start with "r" means raw string

print(re.search(ptn, "dog runs to cat"))    
# <_sre.SRE_Match object; span=(4, 7), match='run'>
```

同样, 中括号 `[]` 中还可以是以下这些或者是这些的组合. 比如 `[A-Z]` 表示的就是所有大写的英文字母. `[0-9a-z]` 表示可以是数字也可以是任何小写字母.

```Python
print(re.search(r"r[A-Z]n", "dog runs to cat"))     
# None

print(re.search(r"r[a-z]n", "dog runs to cat"))    
# <_sre.SRE_Match object; span=(4, 7), match='run'>

print(re.search(r"r[0-9]n", "dog r2ns to cat"))     
# <_sre.SRE_Match object; span=(4, 7), match='r2n'>

print(re.search(r"r[0-9a-z]n", "dog runs to cat"))  
# <_sre.SRE_Match object; span=(4, 7), match='run'>
```

- - -

# 类型匹配

- `\d` : 任何数字
- `\D` : 不是数字
- `\s` : 任何 `white space, 如 [\t\n\r\f\v]`
- `\S` : 不是 `white space`
- `\w` : 任何大小写字母, 数字和 “” [a-zA-Z0-9]
- `\W` : 不是 `\w`
- `\b` : 空白字符 (只在某个字的开头或结尾)
- `\B` : 空白字符 (不在某个字的开头或结尾)
- `\\` : 匹配 \
- `.` : 匹配任何字符 (除了 `\n`)
- `^` : 匹配开头
- `$` : 匹配结尾
- `?` : 前面的字符可有可无

```Python
# \d : decimal digit
print(re.search(r"r\dn", "run r4n"))           
# <_sre.SRE_Match object; span=(4, 7), match='r4n'>

# \D : any non-decimal digit
print(re.search(r"r\Dn", "run r4n"))           
# <_sre.SRE_Match object; span=(0, 3), match='run'>

# \s : any white space [\t\n\r\f\v]
print(re.search(r"r\sn", "r\nn r4n"))          
# <_sre.SRE_Match object; span=(0, 3), match='r\nn'>

# \S : opposite to \s, any non-white space
print(re.search(r"r\Sn", "r\nn r4n"))          
# <_sre.SRE_Match object; span=(4, 7), match='r4n'>

# \w : [a-zA-Z0-9_]
print(re.search(r"r\wn", "r\nn r4n"))          
# <_sre.SRE_Match object; span=(4, 7), match='r4n'>

# \W : opposite to \w
print(re.search(r"r\Wn", "r\nn r4n"))          
# <_sre.SRE_Match object; span=(0, 3), match='r\nn'>

# \b : empty string (only at the start or end of the word)
print(re.search(r"\bruns\b", "dog runs to cat"))    
# <_sre.SRE_Match object; span=(4, 8), match='runs'>

# \B : empty string (but not at the start or end of a word)
print(re.search(r"\B runs \B", "dog   runs  to cat"))  
# <_sre.SRE_Match object; span=(8, 14), match=' runs '>

# \\ : match \
print(re.search(r"runs\\", "runs\ to me"))     
# <_sre.SRE_Match object; span=(0, 5), match='runs\\'>

# . : match anything (except \n)
print(re.search(r"r.n", "r[ns to me"))        
# <_sre.SRE_Match object; span=(0, 3), match='r[n'>

# ^ : match line beginning
print(re.search(r"^dog", "dog runs to cat"))   
# <_sre.SRE_Match object; span=(0, 3), match='dog'>

# $ : match line ending
print(re.search(r"cat$", "dog runs to cat"))   
# <_sre.SRE_Match object; span=(12, 15), match='cat'>

# ? : may or may not occur
print(re.search(r"Mon(day)?", "Monday"))       
# <_sre.SRE_Match object; span=(0, 6), match='Monday'>
print(re.search(r"Mon(day)?", "Mon"))          
# <_sre.SRE_Match object; span=(0, 3), match='Mon'>

```

如果一个字符串有很多行, 我们想使用 `^` 形式来匹配行开头的字符, 如果用通常的形式是不成功的. 比如下面的 `“I”` 出现在第二行开头, 但是使用 `r"^I"` 却匹配不到第二行, 这时候, 我们要使用 另外一个参数, 让 `re.search()` 可以对每一行单独处理. 这个参数就是 `flags=re.M`, 或者这样写也行 `flags=re.MULTILINE`.

```Python
string = """
dog runs to cat.
I run to dog.
"""
print(re.search(r"^I", string))                 
# None

print(re.search(r"^I", string, flags=re.M))     
# <_sre.SRE_Match object; span=(18, 19), match='I'>

```



- - -

# 重复匹配

如果我们想让某个规律被重复使用, 在正则里面也是可以实现的, 而且实现的方式还有很多. 具体可以分为这三种:

- * : 重复零次或多次
- + : 重复一次或多次
- {n, m} : 重复 n 至 m 次
- {n} : 重复 n 次

```Python
# * : occur 0 or more times

print(re.search(r"ab*", "a"))             
# <_sre.SRE_Match object; span=(0, 1), match='a'>

print(re.search(r"ab*", "abbbbb"))        
# <_sre.SRE_Match object; span=(0, 6), match='abbbbb'>

# + : occur 1 or more times

print(re.search(r"ab+", "a"))             
# None

print(re.search(r"ab+", "abbbbb"))        
# <_sre.SRE_Match object; span=(0, 6), match='abbbbb'>

# {n, m} : occur n to m times

print(re.search(r"ab{2,10}", "a"))        
# None

print(re.search(r"ab{2,10}", "abbbbb"))   
# <_sre.SRE_Match object; span=(0, 6), match='abbbbb'>
```
- - -

# Group

我们可以使用group(num) 或 groups() 匹配对象函数来获取匹配表达式。

通过分组, 我们能轻松定位所找到的内容。比如在这个 (\d+) 组里, 需要找到的是一些数字, 在 (.+) 这个组里, 我们会找到 “Date: “ 后面的所有内容。 当使用 match.group() 时, 他会返回所有组里的内容, 而如果给 .group(2) 里加一个数, 它就能定位你需要返回哪个组里的信息。

```Python
match = 
	re.search(r"(\d+), Date: (.+)", 
	"ID: 021523, Date: Feb/12/2017")
print(match.group())                   
# 021523, Date: Feb/12/2017

print(match.group(1))                  
# 021523

print(match.group(2))                  
# Date: Feb/12/2017
```

有时候, 组会很多, 光用数字可能比较难找到自己想要的组, 这时候, 如果有一个名字当做索引, 会是一件很容易的事. 我们字需要在括号的开头写上这样的形式 ?P<名字> 就给这个组定义了一个名字. 然后就能用这个名字找到这个组的内容.

```Python
match = 
	re.search(r"(?P<id>\d+), Date: (?P<date>.+)", 
	"ID: 021523, Date: Feb/12/2017")
print(match.group('id'))                
# 021523
print(match.group('date'))              
# Date: Feb/12/2017
```

- - -

# 检索和替换

`re.sub(pattern, repl, string, count=0, flags=0)`

参数：

- pattern : 正则中的模式字符串。
- repl : 替换的字符串，也可为一个函数。
- string : 要被查找替换的原始字符串。
- count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。

```Python
print(re.sub(r"r[au]ns", "catches", "dog runs to cat"))     
# dog catches to cat

# 其中repl也可以是一个函数，比如

# 将匹配的数字乘以 2
def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)
 
s = 'A23G4HFD567'
print(re.sub('(?P<value>\d+)', double, s))

# A46G8HFD1134

```
 
---

# 编译正则表达式

compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数重复使用。

`re.compile(pattern[, flags])`

- pattern : 一个字符串形式的正则表达式

- flags : 可选，表示匹配模式，比如忽略大小写，多行模式等，可以用`|`操作符表示为同时生效，如`re.I|re.M`，具体参数为：

	- re.I 忽略大小写
	- re.L 表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境
	- re.M 多行模式
	- re.S 即为 . 并且包括换行符在内的任意字符（. 不包括换行符）
	- re.U 表示特殊字符集 \w, \W, \b, \B, \d, \D, \s, \S 依赖于 Unicode 字符属性数据库
	- re.X 为了增加可读性，忽略空格和 # 后面的注释

```Python
compiled_re = re.compile(r"r[ua]n")
print(compiled_re.search("dog ran to cat"))  
# <_sre.SRE_Match object; span=(4, 7), match='ran'>
```

---

# findall

在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。

注意： match 和 search 是匹配一次 findall 匹配所有。

`findall(string[, pos[, endpos]])`

参数：
- string : 待匹配的字符串。
- pos : 可选参数，指定字符串的起始位置，默认为 0。
- endpos : 可选参数，指定字符串的结束位置，默认为字符串的长度。

```Python
# findall
print(re.findall(r"r[ua]n", "run ran ren"))    
# ['run', 'ran']

# | : or
print(re.findall(r"(run|ran)", "run ran ren")) 
# ['run', 'ran']
```
 - - -

# split

split 方法按照能够匹配的子串将字符串分割后返回列表，它的使用形式如下：

`re.split(pattern, string[, maxsplit=0, flags=0])`

参数	|	描述
---|---
pattern	|	匹配的正则表达式
string	|	要匹配的字符串。
maxsplit	|	分隔次数，maxsplit=1 分隔一次，默认为 0，不限制次数。
flags	|	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。

```Python
print(re.split(r"[,;\.]", "a;b,c.d;e")) 
# ['a', 'b', 'c', 'd', 'e']

```
---

# 总结图 

![](http://ope2etmx1.bkt.clouddn.com/regEx.png)
