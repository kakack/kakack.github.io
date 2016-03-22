---

layout: post
categories: [J2EE]
tags: [Java,J2EE,Hibernater]

---


Hibernate是一个开放源代码的对象关系映射框架，它对JDBC进行了非常轻量级的对象封装，使得Java程序员可以随心所欲的使用对象编程思维来操纵数据库。 Hibernate可以应用在任何使用JDBC的场合，既可以在Java的客户端程序使用，也可以在Servlet/JSP的Web应用中使用


####Hibernate中get和load有什么不同之处? 
get和load的最大区别是，如果在缓存中没有找到相应的对象，get将会直接访问数据库并返回一个完全初始化好的对象，而这个过程有可能会涉及到多个数据库调用；而load方法在缓存中没有发现对象的情况下，只会返回一个代理对象，只有在对象getId()之外的其它方法被调用时才会真正去访问数据库，这样就能在某些情况下大幅度提高性能。

####Hibernate中save、persist和saveOrUpdate这三个方法的不同之处？
所有这三个方法，也就是save()、saveOrUpdate()和persist()都是用于将对象保存到数据库中的方法，但其中有些细微的差别。例如，save()只能INSERT记录，但是saveOrUpdate()可以进行 记录的INSERT和UPDATE。还有，save()的返回值是一个Serializable对象，而persist()方法返回值为void。

####Hibernate中的命名SQL查询指的是什么? 
Hibernate的这个面试问题同Hibernate提供的查询功能相关。命名查询指的是用<sql-query>标签在影射文档中定义的SQL查询，可以通过使用Session.getNamedQuery()方法对它进行调用。命名查询使你可以使用你所指定的一个名字拿到某个特定的查询。 Hibernate中的命名查询可以使用注解来定义，也可以使用我前面提到的xml影射问句来定义。在Hibernate中，@NameQuery用来定义单个的命名查询，@NameQueries用来定义多个命名查询。 


####Hibernate中的SessionFactory有什么作用? SessionFactory是线程安全的吗？ 


SessionFactory接口负责初始化Hibernate。它充当数据存储源的代理，并负责创建Session对象。这里用到了工厂模式。需要注意的是SessionFactory并不是轻量级的，因为一般情况下，一个项目通常只需要一个SessionFactory就够，当需要操作多个数据库时，可以为每个数据库指定一个SessionFactory。


顾名思义，SessionFactory就是一个用于创建Hibernate的Session对象的工厂。SessionFactory通常是在应用启动时创建好的，应用程序中的代码用它来获得Session对象。作为一个单个的数据存储，它也是 线程安全的，所以多个线程可同时使用同一个SessionFactory。Java JEE应用一般只有一个SessionFactory，服务于客户请求的各线程都通过这个工厂来获得Hibernate的Session实例，这也是为什么SessionFactory接口的实现必须是线程安全的原因。还有，SessionFactory的内部状态包含着同对象关系影射有关的所有元数据，它是不可变的，一旦创建好后就不能对其进行修改了。

####Hibernate中的Session指的是什么? 可否将单个的Session在多个线程间进行共享？

类似于JDBC connection，Session接口负责执行被持久化对象的CRUD操作(CRUD的任务是完成与数据库的交流，包含了很多常见的SQL语句)。但需要注意的是Session对象是非线程安全的。同时，Hibernate的session不同于JSP应用中的HttpSession。这里当使用session这个术语时，其实指的是Hibernate中的session，而以后会将HttpSession对象称为用户session。

常用的操作：save(),delete(),update(),SaveOrUpdate(),load(),get()

Java状态：临时(Transient：无关联session)、持久化（persistent：关联了session）、游离(Detached：无关联session但有DB record)

 
Session代表着Hibernate所做的一小部分工作，它负责维护着同数据库的链接而且不是线程安全的，也就是说，Hibernage中的Session不能在多个线程间进行共享。虽然Session会以主动滞后的方式获得数据库连接，但是Session最好还是在用完之后立即将其关闭。 

####hibernate中sorted collection和ordered collection有什么不同? 
这个是你会碰到的所有Hibernate面试问题中比较容易的问题。sorted collection是通过使用Java的Comparator在内存中进行排序的，ordered collection中的排序用的是数据库的order by子句。对于比较大的数据集，为了避免在内存中对它们进行排序而出现 Java中的OutOfMemoryError，最好使用ordered collection。

####Hibernate中transient、persistent、detached对象三者之间有什么区别？ 
在Hibernate中，对象具有三种状态：transient、persistent和detached。

同Hibernate的session有关联的对象是persistent对象。对这种对象进行的所有修改都会按照事先设定的刷新策略，反映到数据库之中，也即，可以在对象的任何一个属性发生改变时自动刷新，也可以通过调用Session.flush()方法显式地进行刷新。

如果一个对象原来同Session有关联关系，但当下却没有关联关系了，这样的对象就是detached的对象。你可以通过调用任意一个session的update()或者saveOrUpdate()方法，重新将该detached对象同相应的seesion建立关联关系。

Transient对象指的是新建的持久化类的实例，它还从未同Hibernate的任何Session有过关联关系。同样的，你可以调用persist()或者save()方法，将transient对象变成persistent对象。

####Hibernate中Session的lock()方法有什么作用? 
这是一个比较棘手的Hibernate面试问题，因为Session的lock()方法**重建了关联关系却并没有同数据库进行同步和更新**。因此，你在使用lock()方法时一定要多加小心。顺便说一下，在进行关联关系重建时，你可以随时使用Session的update()方法同数据库进行同步。有时这个问题也可以这么来问：Session的lock()方法和update()方法之间有什么区别？。

####Hibernate中二级缓存指的是什么？ 
二级缓存是在SessionFactory这个级别维护的缓存，它能够通过节省几番数据库调用往返来提高性能。还有一点值得注意，二级缓存是针对整个应用而不是某个特定的session的。


Hibernate 中提供了两级Cache（高速缓冲存储器），第一级别的缓存是Session级别的缓存，它是属于事务范围的缓存。这一级别的缓存由hibernate管理的，一般情况下无需进行干预；第二级别的缓存是SessionFactory级别的缓存，它是属于进程范围或群集范围的缓存。这一级别的缓存可以进行配置和更改，并且可以动态加载和卸载。 Hibernate还为查询结果提供了一个查询缓存，它依赖于第二级缓存。

- 一级缓存:当应用程序调用Session的save()、update()、saveOrUpdate()、get()或load()，以及调用查询接口的 list()、iterate()或filter()方法时，如果在Session缓存中还不存在相应的对象，Hibernate就会把该对象加入到第一级缓存中。当清理缓存时，Hibernate会根据缓存中对象的状态变化来同步更新数据库。 Session为应用程序提供了两个管理缓存的方法：evict(Object obj)：从缓存中清除参数指定的持久化对象。 clear()：清空缓存中所有持久化对象。
- 二级缓存:一般策略为：
	1) 条件查询的时候，总是发出一条select * from table_name where …. （选择所有字段）这样的SQL语句查询数据库，一次获得所有的数据对象。
	2) 把获得的所有数据对象根据ID放入到第二级缓存中。
	3) 当Hibernate根据ID访问数据对象的时候，首先从Session一级缓存中查；查不到，如果配置了二级缓存，那么从二级缓存中查；查不到，再查询数据库，把结果按照ID放入到缓存。
	4) 删除、更新、增加数据的时候，同时更新缓存。

适合放在二级缓存中的数据： 1 很少被修改的数据 2 不是很重要的数据，允许出现偶尔并发的数据 3 不会被并发访问的数据 4 参考数据,指的是供应用参考的常量数据，它的实例数目有限，它的实例会被许多其他类的实例引用，实例极少或者从来不会被修改。

不适合存放到第二级缓存的数据：1 经常被修改的数据 2 财务数据，绝对不允许出现并发 3 与其他应用共享的数据。

Hibernate的二级缓存是一个插件：

- EhCache：可作为进程范围的缓存，存放数据的物理介质可以是内存或硬盘，对Hibernate的查询缓存提供了支持。
- OSCache：可作为进程范围的缓存，存放数据的物理介质可以是内存或硬盘，提供了丰富的缓存数据过期策略，对Hibernate的查询缓存提供了支持。
- SwarmCache：可作为群集范围内的缓存，但不支持Hibernate的查询缓存。
- JBossCache：可作为群集范围内的缓存，支持事务型并发访问策略，对Hibernate的查询缓存提供了支持。

Hibernate的二级缓存策略，是针对于ID查询的缓存策略，对于条件查询则毫无作用。为此，Hibernate提供了针对条件查询的Query Cache（见下）。



####Hibernate中的查询缓存指的是什么？ 
查询缓存实际上保存的是sql查询的结果，这样再进行相同的sql查询就可以之间从缓存中拿到结果了。为了改善性能，查询缓存可以同二级缓存一起来使用。Hibernate支持用多种不同的开源缓存方案，比如EhCache，来实现查询缓存。

查询缓存主要是针对普通属性结果集的缓存， 而对于实体对象的结果集只缓存id。在一级缓存,二级缓存和查询缓存都打开的情况下作查询操作时这样的：查询普通属性，会先到查询缓存中取，如果没有，则查询数据库；查询实体，会先到查询缓存中取id，如果有，则根据id到缓存(一级/二级)中取实体，如果缓存中取不到实体，再查询数据库。

####为什么在Hibernate的实体类中要提供一个无参数的构造器这一点非常重要？
每个Hibernate实体类必须包含一个 无参数的构造器, 这是因为Hibernate框架要使用Reflection API，通过调用Class.newInstance()来创建这些实体类的实例。如果在实体类中找不到无参数的构造器，这个方法就会抛出一个InstantiationException异常。


####可不可以将Hibernate的实体类定义为final类? 
是的，你可以将Hibernate的实体类定义为final类，但这种做法并不好。因为Hibernate会使用代理模式在延迟关联的情况下提高性能，如果你把实体类定义成final类之后，因为 Java不允许对final类进行扩展，所以Hibernate就无法再使用代理了，如此一来就限制了使用可以提升性能的手段。不过，如果你的持久化类实现了一个接口而且在该接口中声明了所有定义于实体类中的所有public的方法轮到话，你就能够避免出现前面所说的不利后果。 



优点：
1、封装了jdbc，简化了很多重复性代码。
2、简化了DAO层编码工作，使开发更对象化了。
3、移植性好，支持各种数据库，如果换个数据库只要在配置文件中变换配置就可以了，不用改变hibernate代码。
4、支持透明持久化，因为hibernate操作的是纯粹的（pojo）java类，没有实现任何接口，没有侵入性。所以说它是一个轻量级框架。

优化：
1、使用一对多的双向关联，尽量从多的一端维护。
2、不要使用一对一，尽量使用多对一。
3、配置对象缓存，不要使用集合缓存。
4、表字段要少，表关联不要怕多，有二级缓存撑腰。
