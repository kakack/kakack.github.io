---

layout: post
categories: [Java]
tags: [Java, Thread]

---

# Java中`Volatile`变量的应用

`Volatile`变量总体上算作一种轻量级的`synchronized`，只是开销少，代码少，功能受到一定的阉割。

锁提供两种特性：
- 互斥：一次只允许一个线程持有某个特定的锁
- 可见性：确保释放锁之前对共享的数据做出的变更对于随后获得该锁的线程可见。

`Volatile`变量具有可见性但不具有原子性。不会像某些锁一样造成线程阻塞，很少有伸缩性问题，如果读操作大大多于写操作，性能上甚至更优。

---

## 使用`Volatile`的条件

- 对变量的写操作不依赖于当前值
- 该变量不包含在具有其他变量的不变式中

第一个限制使得`volatile`变量不能作线程安全计数器。

以下代码显示了一个非线程安全的数值范围，包含了一个不变式——`lower`总是不大于`upper`

```java
@NotThreadSafe 
public class NumberRange {
    private int lower, upper;

    public int getLower() { return lower; }
    public int getUpper() { return upper; }

    public void setLower(int value) { 
        if (value > upper) 
            throw new IllegalArgumentException(...);
        lower = value;
    }

    public void setUpper(int value) { 
        if (value < lower) 
            throw new IllegalArgumentException(...);
        upper = value;
    }
}
```

这种方式限制了范围的状态变量，因此将 lower 和 upper 字段定义为 volatile 类型不能够充分实现类的线程安全；从而仍然需要使用同步。否则，如果凑巧两个线程在同一时间使用不一致的值执行 setLower 和 setUpper 的话，则会使范围处于不一致的状态。例如，如果初始状态是 (0, 5)，同一时间内，线程 A 调用 setLower(4) 并且线程 B 调用 setUpper(3)，显然这两个操作交叉存入的值是不符合条件的，那么两个线程都会通过用于保护不变式的检查，使得最后的范围值是 (4, 3) —— 一个无效值。至于针对范围的其他操作，我们需要使 setLower() 和 setUpper() 操作原子化 —— 而将字段定义为 volatile 类型是无法实现这一目的的。

---

## 使用`Volatile`的模式

#### 1，状态标识

使用一个`boolean`变量用于指示一个重要的一次性事件，如：

```java
volatile boolean shutdownRequested;
...
public void shutdown() { shutdownRequested = true; }

public void doWork() { 
    while (!shutdownRequested) { 
        // do stuff
    }
}
```

可能从外部调用`shutdown()`方法，控制标志转换

### 2， 一次性安全发布 one-time safe publication

同步使得某个更新对象引用从另一个线程写入，确保可见性。
```java
public class BackgroundFloobleLoader {
    public volatile Flooble theFlooble;

    public void initInBackground() {
        // do lots of stuff
        theFlooble = new Flooble();  // this is the only write to theFlooble
    }
}

public class SomeOtherClass {
    public void doWork() {
        while (true) { 
            // do some stuff...
            // use the Flooble, but only if it is ready
            if (floobleLoader.theFlooble != null) 
                doSomething(floobleLoader.theFlooble);
        }
    }
}

```

### 3，独立观察

定期 “发布” 观察结果供程序内部使用

```java

public class UserManager {
    public volatile String lastUser;

    public boolean authenticate(String user, String password) {
        boolean valid = passwordIsValid(user, password);
        if (valid) {
            User u = new User();
            activeUsers.add(u);
            lastUser = user;
        }
        return valid;
    }
}
```

### 4，`Volatile bean`模式

很多框架为易变数据的持有者（例如 HttpSession）提供了容器，但是放入这些容器中的对象必须是线程安全的。

```java

@ThreadSafe
public class Person {
    private volatile String firstName;
    private volatile String lastName;
    private volatile int age;

    public String getFirstName() { return firstName; }
    public String getLastName() { return lastName; }
    public int getAge() { return age; }

    public void setFirstName(String firstName) { 
        this.firstName = firstName;
    }

    public void setLastName(String lastName) { 
        this.lastName = lastName;
    }

    public void setAge(int age) { 
        this.age = age;
    }
}
```

### 5，开销较低的读写锁模式

如果读操作远远超过写操作，可以结合使用内部锁和`volatile`变量来减少公共代码路径的开销。

```java
@ThreadSafe
public class CheesyCounter {
    // Employs the cheap read-write lock trick
    // All mutative operations MUST be done with the 'this' lock held
    @GuardedBy("this") private volatile int value;

    public int getValue() { return value; }

    public synchronized int increment() {
        return value++;
    }
}
```