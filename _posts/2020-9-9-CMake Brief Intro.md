---

layout: post
categories: [C++]
tags: [C++]
title: CMake Brief Intro
date: 2020-09-09
author: Kaka Chen
comments: true
toc: true
pinned: true

---

# 安装

- OS: MacOS Catalina

可以直接在[`cmake`官网](https://cmake.org/download/)上下载含GUI的app，直接本地安装后，用以下三种方法安装`Command Line Use`：

- One may add CMake to the PATH: `PATH="/Applications/CMake.app/Contents/bin":"$PATH"`
- Or, to install symlinks to '/usr/local/bin', run: `sudo "/Applications/CMake.app/Contents/bin/cmake-gui" --install`
- Or, to install symlinks to another directory, run: `sudo "/Applications/CMake.app/Contents/bin/cmake-gui" --install=/path/to/bin`

# 组件描述
在 linux 平台下使用 CMake 生成 Makefile 并编译的流程如下：

> 1 - 编写CMake配置文件`CMakeLists.txt`。  
> 2 - 执行命令`cmake PATH`或者`ccmake PATH`生成`Makefile`（`ccmake`和`cmake`的区别在于前者提供了一个交互式的界面）。其中， `PATH` 是`CMakeLists.txt`所在的目录。  
> 3 - 使用`make`命令进行编译。

针对当前源文件`main.cpp`编写`CMakeLists.txt`:

```CMake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.18.2)

# 项目信息
project (CMakeTutorial)

# 指定生成目标
add_executable(CMakeTutorial main.cpp)
```

`CMakeLists.txt`的语法比较简单，由命令、注释和空格组成，其中命令是不区分大小写的。符号 # 后面的内容被认为是注释。命令由命令名称、小括号和参数组成，参数之间使用空格进行间隔。

对于上面的`CMakeLists.txt`文件，依次出现了几个命令：

> - `cmake_minimum_required`：指定运行此配置文件所需的`CMake`的最低版本；
> - `project`：参数值是`CMakeTutorial`，该命令表示项目的名称是`CMakeTutorial`。
> - `add_executable`： 将名为`main.cpp`的源文件编译成一个名称为`CMakeTutorial`的可执行文件。

# 编译项目

在当前文件夹下直接执行`cmake .`命令，可以得到`Makefile`文件，然后执行`make`命令即可编译当前`main.cpp`成可执行文件`CMakeTutorial`

```bash
(base) ➜  CMakeTutorial git:(master) ✗ cmake .
-- The C compiler identification is AppleClang 11.0.3.11030032
-- The CXX compiler identification is AppleClang 11.0.3.11030032
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/kakack/Documents/Cpp_Primer_Practice/CMakeTutorial
(base) ➜  CMakeTutorial git:(master) ✗ make
Scanning dependencies of target CMakeTutorial
[ 50%] Building CXX object CMakeFiles/CMakeTutorial.dir/main.cpp.o
[100%] Linking CXX executable CMakeTutorial
[100%] Built target CMakeTutorial
```

# 添加多个文件

现在进一步将`MathFunctions.hpp`和`MathFunctions.cpp`文件移动到`math`目录下。

```
./CMakeTutorial
    |
    + - main.cpp
    |
    + - math/
          |
          + -  MathFunctions.cpp
          |
          + -  MathFunctions.hpp
```


对于这种情况，需要分别在项目根目录`CMakeTutorial`和`math`目录里各编写一个`CMakeLists.txt`文件。为了方便，我们可以先将`math`目录里的文件编译成静态库再由`main`函数调用。

根目录中的`CMakeLists.txt`：

```CMake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.18.2)

# 项目信息
project (CMakeTutorial)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)

# 添加 math 子目录
add_subdirectory(math)

# 指定生成目标
add_executable(CMakeTutorial main.cpp)

# 添加链接库
target_link_libraries(CMakeTutorial MathFunctions)
```

该文件添加了下面的内容: 使用命令`add_subdirectory`指明本项目包含一个子目录`math`，这样 `math`目录下的`CMakeLists.txt`文件和源代码也会被处理 。使用命令 `target_link_libraries`指明可执行文件需要连接一个名为`MathFunctions`的链接库 。

子目录中的`CMakeLists.txt`：

```CMake
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
aux_source_directory(. DIR_LIB_SRCS)

# 生成链接库
add_library (MathFunctions ${DIR_LIB_SRCS})
```

在该文件中使用命令`add_library`将`src`目录中的源文件编译为静态链接库。


# 自定义编译选项

`CMake`允许为项目增加编译选项，从而可以根据用户的环境和需求选择最合适的编译方案。

例如，可以将`MathFunctions`库设为一个可选的库，如果该选项为`ON`，就使用该库定义的数学函数来进行运算。否则就调用标准库中的数学函数库。

## 修改`CMakeLists`文件

我们要做的第一步是在顶层的`CMakeLists.txt`文件中添加该选项：

```CMake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.18.2)

# 项目信息
project (CMakeTutorial)

# 加入一个配置头文件，用于处理 CMake 对源码的设置
configure_file (
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
  )

# 是否使用自己的 MathFunctions 库
option (USE_MYMATH
       "Use provided math implementation" ON)

# 是否加入 MathFunctions 库
if (USE_MYMATH)
  include_directories ("${PROJECT_SOURCE_DIR}/math")
  add_subdirectory (math)  
  set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
endif (USE_MYMATH)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)

# 指定生成目标
add_executable(CMakeTutorial ${DIR_SRCS})
target_link_libraries (CMakeTutorial  ${EXTRA_LIBS})
```

其中：

- `configure_file`命令用于加入一个配置头文件`config.h`，这个文件由`CMake从`config.h.in`生成，通过这样的机制，将可以通过预定义一些参数和变量来控制代码的生成。
- `option`命令添加了一个`USE_MYMATH`选项，并且默认值为`ON`。
- 根据`USE_MYMATH`变量的值来决定是否使用我们自己编写的`MathFunctions`库。


## 修改`main.cpp`文件

之后修改`main.cc`文件，让其根据`USE_MYMATH`的预定义值来决定是否调用标准库还是`MathFunctions`库：

## 编写 config.h.in 文件

上面的程序引用了一个`config.h`文件，这个文件预定义了`USE_MYMATH`的值。但我们并不直接编写这个文件，为了方便从`CMakeLists.txt`中导入配置，我们编写一个`config.h.in`文件，内容如下

```
#cmakedefine USE_MYMATH

```

这样`CMake`会自动根据`CMakeLists`配置文件中的设置自动生成`config.h`文件。

## 编译项目

现在编译一下这个项目，为了便于交互式的选择该变量的值，可以使用`ccmake`命令（也可以使用`cmake -i`命令，该命令会提供一个会话式的交互式配置界面）。

从中可以找到刚刚定义的`USE_MYMATH`选项，按键盘的方向键可以在不同的选项窗口间跳转，按下 enter 键可以修改该选项。修改完成后可以按下`c`选项完成配置，之后再按`g`键确认生成` Makefile`。 `ccmake`的其他操作可以参考窗口下方给出的指令提示。

我们可以试试分别将`USE_MYMATH`设为`ON`和`OFF`得到的结果：

`USE_MYMATH`为`ON`

运行结果
```bash
(base) ➜  CMakeTutorial git:(master) ✗ ccmake .
(base) ➜  CMakeTutorial git:(master) ✗ make
[ 50%] Built target MathFunctions
Scanning dependencies of target CMakeTutorial
[ 75%] Building CXX object CMakeFiles/CMakeTutorial.dir/main.cpp.o
[100%] Linking CXX executable CMakeTutorial
[100%] Built target CMakeTutorial
(base) ➜  CMakeTutorial git:(master) ✗ ./CMakeTutorial 3 4
Now we use our own Math library.
3 ^ 4 is 81
```

此时`config.h`的内容为：

```C
#define USE_MYMATH
```

`USE_MYMATH`为`OFF`
运行结果：

```bash
(base) ➜  CMakeTutorial git:(master) ✗ ./CMakeTutorial 3 4
Now we use the standard library.
3 ^ 4 is 81
```

此时`config.h`的内容为：

```C
/* #undef USE_MYMATH */
```

# 安装和测试

`CMake`也可以指定安装规则，以及添加测试。这两个功能分别可以通过在产生`Makefile`后使用`make install`和`make test`来执行。在以前的`GNU Makefile`里，你可能需要为此编写 `install`和`test`两个伪目标和相应的规则，但在`CMake`里，这样的工作同样只需要简单的调用几条命令。

首先先在`math/CMakeLists.txt`文件里添加下面两行：

```CMake
# 指定 MathFunctions 库的安装路径
install (TARGETS MathFunctions DESTINATION bin)
install (FILES MathFunctions.hpp DESTINATION include)
```

指明`MathFunctions`库的安装路径。之后同样修改根目录的`CMakeLists`文件，在末尾添加下面几行：

```CMake
# 指定安装路径
install (TARGETS CMakeTutorial DESTINATION bin)
install (FILES "${PROJECT_BINARY_DIR}/config.h"
         DESTINATION include)
```

通过上面的定制，生成的`CMakeTutorial`文件和`MathFunctions`函数库`libMathFunctions.o`文件将会被复制到`/usr/local/bin`中，而`MathFunctions.hpp`和生成的`config.h`文件则会被复制到`/usr/local/include`中。我们可以验证一下（顺带一提的是，这里的`/usr/local/`是默认安装到的根目录，可以通过修改`CMAKE_INSTALL_PREFIX`变量的值来指定这些文件应该拷贝到哪个根目录）：

```bash
(base) ➜  CMakeTutorial git:(master) ✗ make install
[100%] Built target CMakeTutorial
Install the project...
-- Install configuration: ""
-- Installing: /usr/local/bin/CMakeTutorial
-- Installing: /usr/local/include/config.h
(base) ➜  CMakeTutorial git:(master) ✗ ls /usr/local/bin
CMakeTutorial   ccmake          cpack           gettext
(base) ➜  CMakeTutorial git:(master) ✗ ls /usr/local/include
config.h      idn2.h        textstyle     unicase.h
```

# 为工程添加测试

添加测试同样很简单。`CMake`提供了一个称为`CTest`的测试工具。我们要做的只是在项目根目录的`CMakeLists`文件中调用一系列的`add_test`命令。

```CMake
# 启用测试
enable_testing()

# 测试程序是否成功运行
add_test (test_run CMakeTutorial 5 2)

# 测试帮助信息是否可以正常提示
add_test (test_usage CMakeTutorial)
set_tests_properties (test_usage
  PROPERTIES PASS_REGULAR_EXPRESSION "Usage: .* base exponent")

# 测试 5 的平方
add_test (test_5_2 CMakeTutorial 5 2)

set_tests_properties (test_5_2
 PROPERTIES PASS_REGULAR_EXPRESSION "is 25")

# 测试 10 的 5 次方
add_test (test_10_5 CMakeTutorial 10 5)

set_tests_properties (test_10_5
 PROPERTIES PASS_REGULAR_EXPRESSION "is 100000")

# 测试 2 的 10 次方
add_test (test_2_10 CMakeTutorial 2 10)

set_tests_properties (test_2_10
 PROPERTIES PASS_REGULAR_EXPRESSION "is 1024")
```

上面的代码包含了四个测试。第一个测试`test_run`用来测试程序是否成功运行并返回0值。剩下的三个测试分别用来测试 5 的 平方、10 的 5 次方、2 的 10 次方是否都能得到正确的结果。其中`PASS_REGULAR_EXPRESSION`用来测试输出是否包含后面跟着的字符串。

让我们看看测试的结果：

```bash
(base) ➜  CMakeTutorial git:(master) ✗ cmake .
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/kakack/Documents/Cpp_Primer_Practice/CMakeTutorial
(base) ➜  CMakeTutorial git:(master) ✗ make test
Running tests...
Test project /Users/kakack/Documents/Cpp_Primer_Practice/CMakeTutorial
    Start 1: test_run
1/5 Test #1: test_run .........................   Passed    0.00 sec
    Start 2: test_usage
2/5 Test #2: test_usage .......................   Passed    0.00 sec
    Start 3: test_5_2
3/5 Test #3: test_5_2 .........................   Passed    0.00 sec
    Start 4: test_10_5
4/5 Test #4: test_10_5 ........................   Passed    0.00 sec
    Start 5: test_2_10
5/5 Test #5: test_2_10 ........................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =   0.02 sec
```

如果要测试更多的输入数据，像上面那样一个个写测试用例未免太繁琐。这时可以通过编写宏来实现：

```
# 定义一个宏，用来简化测试工作
macro (do_test arg1 arg2 result)
  add_test (test_${arg1}_${arg2} CMakeTutorial ${arg1} ${arg2})
  set_tests_properties (test_${arg1}_${arg2}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result})
endmacro (do_test)
 
# 使用该宏进行一系列的数据测试
do_test (5 2 "is 25")
do_test (10 5 "is 100000")
do_test (2 10 "is 1024")
```

关于`CTest`的更详细的用法可以通过`man 1 ctest`参考`CTest`的文档。


# 支持 gdb
让`CMake`支持`gdb`的设置也很容易，只需要指定`Debug`模式下开启`-g`选项：

```bash
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
```

之后可以直接对生成的程序使用`gdb`来调试。

# 添加环境检查

有时候可能要对系统环境做点检查，例如要使用一个平台相关的特性的时候。在这个例子中，我们检查系统是否自带`pow`函数。如果带有`pow`函数，就使用它；否则使用我们定义的`power`函数。

## 添加`CheckFunctionExists`宏

首先在顶层`CMakeLists`文件中添加`CheckFunctionExists.cmake`宏，并调用`check_function_exists`命令测试链接器是否能够在链接阶段找到`pow`函数。

```C
# 检查系统是否支持 pow 函数
include (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
check_function_exists (pow HAVE_POW)
```

将上面这段代码放在`configure_file`命令前。

## 预定义相关宏变量

接下来修改`config.h.in`文件，预定义相关的宏变量。

```CMake
// does the platform provide pow function?
#cmakedefine HAVE_POW
```

## 在代码中使用宏和函数

最后一步是修改`main.cpp`，在代码中使用宏和函数：

```C
#ifdef HAVE_POW
    printf("Now we use the standard library. \n");
    double result = pow(base, exponent);
#else
    printf("Now we use our own Math library. \n");
    double result = power(base, exponent);
#endif
```

# 添加版本号

给项目添加和维护版本号是一个好习惯，这样有利于用户了解每个版本的维护情况，并及时了解当前所用的版本是否过时，或是否可能出现不兼容的情况。

首先修改顶层`CMakeLists`文件，在`project`命令之后加入如下两行：

```
set (CMakeTutorial_VERSION_MAJOR 1)
set (CMakeTutorial_VERSION_MINOR 0)
```

分别指定当前的项目的主版本号和副版本号。

之后，为了在代码中获取版本信息，我们可以修改`config.h.in`文件，添加两个预定义变量：

```C
// the configured options and settings for Tutorial
#define CMakeTutorial_VERSION_MAJOR @CMakeTutorial_VERSION_MAJOR@
#define CMakeTutorial_VERSION_MINOR @CMakeTutorial_VERSION_MINOR@
```

这样就可以直接在代码中打印版本信息了


