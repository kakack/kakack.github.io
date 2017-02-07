---

layout: post
categories: [Hadoop]
tags: [Hadoop, Apache, Avro]

---

Apache Avro是一个数据序列化系统，提供：

- 丰富的数据结构
- 一个紧凑的、快速的、二进制数据格式
- 一个容器文件，保存持久化数据
- 远程生产调用（Remote producer call，RPC）
- 和动态语言简单集成

## Schema

Avro依赖于schema，当读写时都是以此形式存在，不需要其他额外的标识，使得序列化快捷简单，可以理解为Java中的类。整个schema是fully self-describing的，当数据保存在文件中的时候，与数据本身一并保存。当Avro远程调用时，客户端和服务器端用握手协议交换schema。Avro Schema是用json形式定义。

## 与其他系统相比

区别在于

- *动态输入* ： 不需要每次都生成代码，数据一致都伴随着schema，比如Protocol Buffers需要用户先定义好数据结构，然后根据这个结构生成代码，再组装数据，如果数据来自多个数据源，就需要重复执行上述操作。
- *不加标签的数据* ：由于schema一致伴随着数据，所以期望数据中不要保留太多冗余信息，以获取更小的序列化尺寸。
- *不需要手动分配field IDs* ： 当schema改变时，不管是新的还是旧的schema都会在处理数据时存在，所以可以用field name来象征性地解决新旧schema之间的冲突。

- - -

## Get Started

##### 1, 下载Avro的jar包

从[Avro官网](http://avro.apache.org/releases.html)下载所需的jar包，当前最新版本是1.8.1，我在这只用了1.7.7版本，下载了`avro-1.7.7.jar`和`avro-tools-1.7.7.jar`两个包，放在项目的lib文件夹中。

##### 2，定义Schema

生成文件`user.avsc`

```json
{"namespace": "com.Avro",
 "type": "record",
 "name": "User",
 "fields": [
     {"name": "name", "type": "string"},
     {"name": "favorite_number",  "type": ["int", "null"]},
     {"name": "favorite_color", "type": ["string", "null"]}
 ]
}

```

##### 3, 编译Schema

执行：

`java -jar ${项目路径}/src/lib/avro-tools-1.7.7.jar compile schema user.avsc .`

在当前路径下生成`com/avro/User.java`目录和文件。

##### 4，编写测试文件

```java
public class AvroTest {

	//序列化
    @Test
    public void Serialization() throws IOException {
        User user1 = new User();
        user1.setName("Huyifan");
        user1.setFavoriteColor("Yello");
        user1.setFavoriteNumber(1024);

        User user2 = new User("Pengdameng", 12, "Red");

        User user3 = User.newBuilder()
                .setName("Xidongdong")
                .setFavoriteColor("Blue")
                .setFavoriteNumber(32).build();
        
        //三种不同的方法构建三个User对象

        String path = "/Users/apple/Personal/GitKaka/HadoopExample/data/avro/user(1).avsc";
        DatumWriter<User> userDatumWriter = new SpecificDatumWriter<User>(User.class);
        DataFileWriter<User> dataFileWriter = new DataFileWriter<User>(userDatumWriter);
        dataFileWriter.create(user1.getSchema(), new File(path));

        dataFileWriter.append(user1);
        dataFileWriter.append(user2);
        dataFileWriter.append(user3);
        dataFileWriter.close();
        
        //序列化user1、user2、user3到文件中，目标文件名为user(1).avsc

    }

    //反序列化
    @Test
    public void Deserialization() throws IOException {
        DatumReader<User> reader = new SpecificDatumReader<User>(User.class);
        DataFileReader<User> dataFileReader =
                new DataFileReader<User>(new File("/Users/apple/Personal/GitKaka/HadoopExample/data/avro/user(1).avsc"), reader);
                
        //初始化reader
                
        User user = null;
        while (dataFileReader.hasNext()){
            user = dataFileReader.next();
            System.out.println(user);
        }
    }

    public static void main(String[] args) throws Exception{
        Result result = JUnitCore.runClasses(AssertTest.class);
        for (Failure failure : result.getFailures()) {
            System.out.println(result.wasSuccessful());
        }
    }

}

```