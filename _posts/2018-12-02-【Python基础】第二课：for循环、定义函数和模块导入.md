---
layout:     post
title:      【Python基础】第二课：for循环、定义函数和模块导入
subtitle:   Python语法，for循环，if语句，定义函数，模块导入
date:       2018-12-02
author:     x-jeff
header-img: blogimg/20181202(2).jpg
catalog: true
tags:
    - Python
    - Element Knowledge
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Python语法之for循环，if分支语句
## 1.1.for循环，if分支语句
![](https://ws2.sinaimg.cn/large/006tNbRwly1fxspbt5wmxj30du0h6mxz.jpg)
（%表示取余数。）

需要注意几点：

1. 在c++中会通过{}来表明嵌套关系，限定作用域。而在python中，则通过在语句前面添加空格（或者tab键，相当于四个空格）的方式实现。
2. for语句和if语句的后面都需要有冒号“：”。

如果语句前面没有缩排（空格），python就会认为此语句和之前的for循环没有关系。例如：
![](https://ws3.sinaimg.cn/large/006tNbRwly1fxsplsrnz6j30au0843yu.jpg)

相同缩排的语句在同一结构下，例如：
![](https://ws2.sinaimg.cn/large/006tNbRwly1fxspojlizij30b60b2aaj.jpg)

## 1.2.if,elif,else
![](https://ws4.sinaimg.cn/large/006tNbRwly1fxsql8jatfj30cu08kt9i.jpg)
（python中的逻辑运算符：and、or、not。）

# 2.Python语法之函数
使用`def`定义一个函数：

~~~python
##方法一：
def addNum(a,b) :
	return a+b
addNum(2,3) #输出5

def square(x) : 
	return x*x
square(3) #输出9

##方法二：
addNum = lambda a,b : a+b
addNum(2,3) #输出5

func = lambda x : x**3 #x的3次幂
func(3) #输出27
~~~

`lambda`允许用于快速定义**单行**函数，又称**匿名函数**。

# 3.Python语法之模块导入
Python中库的导入：`import`

~~~python
a=3.2
import math
math.ceil(a) #取大于a的最小整数，所以输出为4

import numpy
numpy.random.normal(25,5,10) #产生符合正态分布，均值为25，标准差为5的10个随机数
~~~