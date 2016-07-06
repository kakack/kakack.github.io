---

layout: post
categories: [Caravel]
tags: [Linux]

---

关于Airbnb Caravel在Mac和Ubuntu系统上安装和使用
---


Caravel是Airbnb新晋开源的一个自助式数据分析和成像工具，用于简化数据库端的分析探索操作，允许操作人员通过创建和修改Dashboard的形式快速获得一个可视化数据的功能。同时还能兼顾数据格式的拓展性、数据模型的高粒度保证、快速的复杂规则查询、兼容主流鉴权模式（数据库、OpenID、LDAP、OAuth或者基于Flask AppBuilder的REMOTE_USER）通过一个定义字段、下拉聚合规则的简单的语法层操作就让我们可以将数据源在U上丰富地呈现。

前置依赖
===
对于caravel的支持只有python 2.7，不支持python3 或者2.6及以下的python版本。

安装过程需要使用pip，所以需要预先安装：

```shell
#For Ubuntu
sudo apt-get install pip

#For Mac
sudo brew install pip
```

对于数据库的访问连接用的是`cryptography`python库。此外，内部用户信息等内容会需要一个mysql存储，所以需要安装mysql和`MySQL-Python`库，操作如下：

```shell
#For Ubuntu
sudo apt-get install build-essential libssl-dev libffi-dev python-dev python-pip

#For Mac
brew install pkg-config libffi openssl python
env LDFLAGS="-L$(brew --prefix openssl)/lib" CFLAGS="-I$(brew --prefix openssl)/include" pip install cryptography
```

安装MySql和MySQL-Python：

```shell
#For Ubuntu
apt-get install mysql

#For Mac
brew install mysql

export PATH=$PATH:/usr/local/mysql/bin
pip install MySQL-Python

```

安装和配置Caravel
===
```shell
# Install caravel
pip install caravel

# Create an admin user
fabmanager create-admin --app caravel

# Initialize the database
caravel db upgrade

# Create default roles and permissions
caravel init

# Load some data to play with
caravel load_examples

# Start the development web server
caravel runserver -d
```
服务启动之后，可以在`http://localhost:8088`上访问页面。如果需要配置自己的应用，需要创建一个文件(module) `caravel_config.py`，放在本地的PythoPath中，例如：

```Python
#---------------------------------------------------------
# Caravel specific config
#---------------------------------------------------------
ROW_LIMIT = 5000
CARAVEL_WORKERS = 16

CARAVEL_WEBSERVER_PORT = 8088
#---------------------------------------------------------

#---------------------------------------------------------
# Flask App Builder configuration
#---------------------------------------------------------
# Your App secret key
SECRET_KEY = '\2\1thisismyscretkey\1\2\e\y\y\h'

# The SQLAlchemy connection string to your database backend
# This connection defines the path to the database that stores your
# caravel metadata (slices, connections, tables, dashboards, ...).
# Note that the connection information to connect to the datasources
# you want to explore are managed directly in the web UI
SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/caravel.db'

# Flask-WTF flag for CSRF
CSRF_ENABLED = True
```
数据库依赖：

|database	|pypi package	|SQLAlchemy URI prefix
|---|---|---|
|MySQL	|pip install mysqlclient	|mysql://
|Postgres	|pip install psycopg2	|postgresql+psycopg2://
|Presto	|pip install pyhive	|presto://
|Oracle	|pip install cx_Oracle	|oracle://
|sqlite	 |	|sqlite://
|Redshift	|pip install sqlalchemy-redshift	|redshift+psycopg2://
|MSSQL	|pip install pymssql	|mssql://



后端
===
整个项目的后端是基于Python的，用到了Flask、Pandas、SqlAlchemy。

- Flask AppBuilder(鉴权、CRUD、规则）
- Pandas（分析）
- SqlAlchemy（数据库ORM）

此外，也关注到Caravel的缓存机制值得我们学习：

- 采用memcache和Redis作为缓存
- 级联超时配置
- UI具有时效性控制
- 允许强制刷新

缺点
===
- Caravel的可视化，目前只支持每次可视化一张表，对于多表join的情况还无能为力
- 依赖于数据库的快速响应，如果数据库本身太慢Caravel也没什么办法
- 语义层的封装还需要完善，因为druid原生只支持部分sql。

参考
===
- [Official site of airbnb caravel](http://airbnb.io/caravel/)
- [[原]解密Airbnb数据流编程神器：Caravel 颠覆许多大数据分析平台的开源工具](https://segmentfault.com/a/1190000005083953)
- [Cavarel github](https://github.com/airbnb/caravel)
- [Cavarel docker image](https://hub.docker.com/r/kochalex/caravel/)
