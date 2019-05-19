---
layout:     post
title:      【Python基础】第五课：读写TXT文件
subtitle:   读txt文件，写txt文件
date:       2019-05-14
author:     x-jeff
header-img: blogimg/20190514.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.写txt文件

```python
f=open("temp.txt","w")
f.write("hello world")
f.close
```

`"w"`执行写操作。上述程序会在相应目录下生成一个`temp.txt`的文件，文件内容为“hello world”。

以下程序可实现自动关闭文档的操作，省去`f.close()`：

```python
with open("temp.txt","w") as f: #冒号不能少
    f.write("hello\nworld")
```

`\n`为[转义序列](http://shichaoxin.com/2019/05/13/C++基础-第七课-字面值常量/)，表示换行。

# 2.读txt文件

python中有三种读取txt文件的方法：

1. `.read()`
2. `.readline()`
3. `.readlines()`

若文本`temp.txt`中的内容为：

```
hello
world
```

分别用三种方式去读其中的内容。

## 2.1.`.read()`

`.read()`每次读取整个文件，它通常用于将文件内容放到**一个字符串变量**中。

使用`.read()`读文件中的内容：

```python
with open("temp.txt","r") as f:
	print(f.read())
```

`"r"`是“只读”操作，上述程序输出为：

```
hello
world
```

其中，

1. `f.read(0)`输出为空字符。
2. `f.read(1)`输出第1个字符，即：

	```
	h
	```
3. `f.read(6)`输出前6个字符，其中第6个字符为换行符，即：

	```
	hello
	```
4. `f.read(7)`输出前7个字符，即：

	```
	hello
	w
	```
5. `f.read(15)`输出的字符个数超过了文件所含的字符个数，并不会报错，会输出该文件的所有字符，即：

	```
	hello
	world
	```

## 2.2.`.readline()`

`.readline()`**每次只读取一行**，通常比`.readlines()`慢很多。仅当没有足够内存可以一次读取整个文件时，才应该使用`.readline()`。返回值也是**一个字符串变量**

```python
with open("temp.txt","r") as f:
    print(f.readline())
```

输出为：

```
hello
```

其中，

1. `f.readline(0)`输出为空字符。
2. `f.readline(1)`输出为`h`。
3. `f.readline(6)`输出为`hello`。
4. `f.readline(7)`输出为`hello`。

上面第3、4个例子可以看出，`.readline()`相当于只读了第一行`hello`，没有读入第二行`world`。

## 2.3.`.readlines()`

`.readlines()`**一次性读取整个文件**，像`.read()`一样。🤜`.readlines()`自动将文件内容分析成一个行的列表🤛。用for...in...处理，返回的是一个**列表结构**。

```python
with open("temp.txt","r") as f:
    print(f.readlines())
```

输出为：`['hello\n', 'world']`。

1. `f.readlines(0)`输出为`['hello\n', 'world']`。
2. `f.readlines(1)`输出为`['hello\n']`。
3. `f.readlines(5)`输出为`['hello\n']`。
4. `f.readlines(6)`输出为`['hello\n', 'world']`。

## 2.4.三种读取方式的比较

👉`.read()`

```python
with open("temp.txt","r") as f:
    for n in f.read():
        print(n)
```

输出为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson5/5x1.png)

👉`.readline()`

```python
with open("temp.txt","r") as f:
    for n in f.readline():
        print(n)
```

输出为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson5/5x2.png)

👉`.readlines()`

```python
with open("temp.txt","r") as f:
    for n in f.readlines():#for n in f: 也可以
        print(n)
```

输出为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson5/5x3.png)

`hello`和`world`中间多了一个空行，这是`.readlines()`的特性，可以通过`.strip()`来删除空行：

```python
with open("temp.txt","r") as f:
    for n in f: 
        print(n.strip())
```

输出为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson5/5x4.png)

# 3.代码地址

1. [读写TXT文件](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo5)


