---
layout:     post
title:      【Python基础】第二十三课：SQLite数据库之数据存储
subtitle:   使用python链接数据库，透过SQLite做数据新增、查询，使用pandas存储数据
date:       2021-07-17
author:     x-jeff
header-img: blogimg/20210717.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.使用python链接数据库

```python
import sqlite3 as lite

con = lite.connect('test.sqlite')#如果不存在test.sqlite，则会新建
cur = con.cursor()
cur.execute('SELECT SQLITE_VERSION()')
data = cur.fetchone()
print(data)#输出版本：('3.26.0',)
con.close()
```

也可以使用`with`来省略`close`操作：

```python
with lite.connect('test.sqlite') as con:
    cur = con.cursor()
    cur.execute('SELECT SQLITE_VERSION()')
    data = cur.fetchone()
    print(data)
```

# 2.透过SQLite做数据新增、查询

```python
with lite.connect('test.sqlite') as con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS PhoneAddress")#会舍弃掉已存在的PhoneAddress表格
    cur.execute(
        "CREATE TABLE PhoneAddress(phone CHAR(10) PRIMARY KEY, address TEXT, name TEXT unique, age INT NOT NULL)")
    cur.execute("INSERT INTO PhoneAddress VALUES('0912173381','United State','Jhon Doe',53)")
    #也可写为：
    #cur.execute("INSERT INTO PhoneAddress(phone,address,name,age) VALUES('0912173381','United State','Jhon Doe',53)")
    cur.execute("INSERT INTO PhoneAddress VALUES('0928375018','Tokyo Japan','MuMu Cat',6)")
    cur.execute("INSERT INTO PhoneAddress VALUES('0957209108','China','Zhang San',29)")
    cur.execute("SELECT phone,address FROM PhoneAddress")
    #取出所有栏位：
    #cur.execute("SELECT * FROM PhoneAddress")
    
    #fetchone一次只取一组数据
    #data = cur.fetchone()#data为：('0912173381', 'United State')
    #fetchall一次取所有数据
    data = cur.fetchall()#data为：[('0912173381', 'United State'), ('0928375018', 'Tokyo Japan'), ('0957209108', 'China')]
    for rec in data:
        print(rec[0], rec[1])
```

输出为：

```
0912173381 United State
0928375018 Tokyo Japan
0957209108 China
```

# 3.使用pandas存储数据

使用pandas的DataFrame批量存入数据。

```python
import sqlite3 as lite
import pandas

# 建立DataFrame
employee = [{'name': 'Mary', 'age': 23, 'gender': 'F'}, {'name': 'John', 'age': 33, 'gender': 'M'}]
df = pandas.DataFrame(employee)
# 使用pandas存储数据
with lite.connect('test.sqlite') as db:
    df.to_sql(name='employee', index=False, con=db, if_exists='replace')
```

# 4.代码地址

1. [SQLite数据库之数据存储](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo23)