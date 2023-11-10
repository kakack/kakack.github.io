---

layout: post
tags: [Python]
title: Python yield关键字
date: 2018-04-17
author: Kaka Chen
comments: true
toc: true
pinned: false


---

# 1. 可迭代对象

Python中有很多可迭代对象，包含`迭代器（iterator）`和`生成器（generator）`，后者就是`yield`关键字所返回的对象。其他可迭代对象包括序列（包括字符串、列表和tuple）和字典。其中所有可迭代对象都必须实现`—__iter__`方法，而所有迭代器都必须实现`__iter__`和`next`方法。

# 迭代器

迭代器是抽象的一个数据流对象，调用`next`方法可以获得被迭代的下一个元素，直到没有元素后抛出`StopIteration`异常。迭代器的 `__iter__()` 方法返回迭代器自身；因此迭代器也是可迭代的。

# 迭代器协议

迭代器协议（iterator protocol）指的是容器类需要包含一个特殊方法。如果一个容器类提供了` __iter__() `方法，并且该方法能返回一个能够逐个访问容器内所有元素的迭代器，则我们说该容器类实现了迭代器协议。

# 生成器函数

与普通函数不同，生成器函数被调用后，其函数体内的代码并不会立即执行，而是返回一个`生成器（generator-iterator）`。当返回的生成器调用成员方法时，相应的生成器函数中的代码才会执行。

```Python
def square():
    for x in range(4):
        yield x ** 2
# square()是一个生成器函数，
# 每次返回后再次调用函数状态都会与从上次返回时一致
square_gen = square()
for x in square_gen:
    print(x)

```

# 生成器的方法

- generator.next()：从上一次在 yield 表达式暂停的状态恢复，继续执行到下一次遇见 yield 表达式。当该方法被调用时，当前 yield 表达式的值为 None，下一个 yield 表达式中的表达式列表会被返回给该方法的调用者。若没有遇到 yield 表达式，生成器函数就已经退出，那么该方法会抛出 StopIterator 异常。
- generator.send(value)：和 generator.next() 类似，差别仅在与它会将当前 yield 表达式的值设置为 value。
- generator.throw(type[, value[, traceback]])：向生成器函数抛出一个类型为 type 值为 value 调用栈为 traceback 的异常，而后让生成器函数继续执行到下一个 yield 表达式。其余行为与 generator.next() 类似。
- generator.close()：告诉生成器函数，当前生成器作废不再使用。 

# 下一个

当调用`generator.next()`时，生成器函数会从当前函数执行到下一个`yield`为止。如

```Python
def next_yield():
    yield 1
    print("first yield")
    yield 2
    print("second yield")
    yield 3
    print("third yield")
#     yield from f123()
m = next_yield()

>>next(m)
# 1
>>next(m)
# first yield
# 2
>>next(m)
# second yield
# 3
```

# 用send传入输入信号量

```Python
def send_yield():
    x = 1
    while True:
        y = (yield x)
        x += y
        print("x = %d, y = %d" %(x, y))
m = send_yield()

next(m)
>>1
m.send(2)
>>x = 3, y = 2
>>3
m.send(3)
>>x = 6, y = 3
>>6
……
```

# 几个常见用法

## yield空

用法类似于循环中的中断器，或者返回一个`none`

```Python
def myfun(total):
    for i in range(total):
        print(i + 1)
        yield
a = myfun(3)
for i in a:
    print(i)

# 1
# None
# 2
# None
# 3
# None

```	

## yield from

从一个特定集合里做`yield`

```Python
def myfun(total):
    for i in range(total):
        yield i
    yield from ['a', 'b', 'c']
    
m = myfun(3)
next(m)
>>0
next(m)
>>1
next(m)
>>2
next(m)
>>'a'
next(m)
>>'b'
next(m)
>>'c'
```

可以用此写一个永不溢出的循环`yield`

```Python
def cycle(p):
    yield from p
    yield from cycle(p)
a = cycle('abcde')
```

## return yield

return只是起到终止函数的作用

```Python
def myfun(total):
    yield from range(total)

>>> a = myfun(4)
>>> a
<generator object myfun at 0x000001B61CCB9CA8>
>>> for i in a:
...     print(i)
...
0
1
2
3
```

这样`a`就是一个生成器

```python
def myfun(total):
	return (yield from range(total))
a = myfun(4)
for i in a:
   print(i)
   
0
1
2
3
```