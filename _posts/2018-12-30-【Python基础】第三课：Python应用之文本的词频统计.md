---
layout:     post
title:      【Python基础】第三课：Python应用之文本的词频统计
subtitle:   词频统计
date:       2018-12-30
author:     x-jeff
header-img: blogimg/20181230.jpg
catalog: true
tags:
    - Python
    - Element Knowledge
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.词频统计（方法一）

首先，我们先输入一段文本：

```python
ai_text='''Knowledge engineering is a core part of AI research. Machines can often act and react like humans only if they have abundant information relating to the world. Artificial intelligence must have access to objects, categories, properties and relations between all of them to implement knowledge engineering. Initiating common sense, reasoning and problem-solving power in machines is a difficult and tedious task.Machine learning is also a core part of AI. Learning without any kind of supervision requires an ability to identify patterns in streams of inputs, whereas learning with adequate supervision involves classification and numerical regressions. Classification determines the category an object belongs to and regression deals with obtaining a set of numerical input or output examples, thereby discovering functions enabling the generation of suitable outputs from respective inputs. Mathematical analysis of machine learning algorithms and their performance is a well-defined branch of theoretical computer science often referred to as computational learning theory.Machine perception deals with the capability to use sensory inputs to deduce the different aspects of the world, while computer vision is the power to analyze visual inputs with a few sub-problems such as facial, object and gesture recognition.Robotics is also a major field related to AI. Robots require intelligence to handle tasks such as object manipulation and navigation, along with sub-problems of localization, motion planning and mapping.'''
```
(文本来源：[https://www.techopedia.com/definition/190/artificial-intelligence-ai](https://www.techopedia.com/definition/190/artificial-intelligence-ai))

现在我们尝试对文本中的字、词进行分割：

```python
ai_text.split() ##默认以空格为分隔符
```

但是`ai_text.split()`只能以空格为分隔符，并不能满足我们分隔字、词的需求，因为文本中分隔符不止空格，还有句号（.）和逗号（,）等。因此我们需要多个分隔符：

```python
import re
re.split('[ .,]',ai_text) ##使用三个分隔符：空格、句号、逗号
```

得到如下分隔结果：

```python
['Knowledge',
 'engineering',
 'is',
 'a',
 'core',
 'part',
 ... ##篇幅原因，中间省略
 'motion',
 'planning',
 'and',
 'mapping',
 '']
```

>列表中的元素可以任意形式，比如数值、字符串、字典、列表等。  
>例如`li=["abc",12,{'a':1},[1,2,3]]`

这里存在一个问题，单词中的字母有的是大写有的是小写，这就会影响我们最后的词频统计结果，因为比如`Word`和`word`会被认为是两个不同的单词，所以我们进行进一步的处理，将字母统一变为小写：

```python
ai_text_split=re.split('[ ,.]',ai_text.lower())
```

接下来我们建立一个空字典用于统计每个词的频次：

```python
dic={} ##建立一个空字典
for word in ai_text_split:
    if word not in dic: ##not in的用法很方便
        dic[word]=1
    else:
        dic[word]=dic[word]+1
```

字典`dic`的内容如下：

```python
{'knowledge': 2,
 'engineering': 2,
 'is': 6,
 'a': 7,
 'core': 2,
 'part': 2,
 ... ##篇幅原因，中间省略
 'navigation': 1,
 'along': 1,
 'localization': 1,
 'motion': 1,
 'planning': 1,
 'mapping': 1}
```

到目前为止，我们已经统计了文本中每个词的词频，但是没有按照词频的大小排列。

>**python中查看某一个包中某一个函数的用法：**
>
>```python
>import scipy
>print scipy.squeeze.__doc__ ##方法一
>print help(scipy.squeeze) ##方法二
>```

在对词频排序之前我们先介绍**operator库**：

* **operator库**：提供了一系列的函数操作。
* **operator.itemgetter函数**：用于获取对象的某一维的数据

**operator.itemgetter函数**的使用：

```python
import operator

students=[('john','A',15),('jane','B',12),('dave','B',10)]
single_stu=('john','A',15)

b=operator.itemgetter(2) ##获取第三维的数据！
b(students) ##输出为：('dave', 'B', 10)
b(single_stu) ##输出为：15

c=operator.itemgetter(2,1) ##获取第三维和第二维的数据！注意顺序！
c(students) ##输出为：(('dave', 'B', 10), ('jane', 'B', 12))
c(single_stu) ##输出为：(15, 'A')
```

然后我们利用`sorted()`函数结合`operator.itemgetter()`函数进行排序：

```python
##例子1：根据第三维进行排序
sorted(students,key=operator.itemgetter(2)) ##输出为：[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]

##例子2：先排第二维的顺序，在第二维顺序正确的基础上再排第三维的顺序
sorted(students,key=operator.itemgetter(1,2)) ##输出为：[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]
```

**tips:***reverse=T or F*可以改变正序或倒序！

`sorted(students,key=operator.itemgetter(2))`可以改写为：`sorted(students,key=lambda x:x[2])`，相当于一个迭代的过程，依次从`students`中抽取元素，`x`依次等于`('john','A',15)`、`('jane','B',12)`和`('dave','B',10)`。

Python字典（Dictionary）`.items()`函数以**列表**返回可遍历的（key,value）元组数组，返回输出为：`dict_items([(key1,value1),(key2,value2),...])`，则`dic.items()`的结果为：

```python
dict_items([('knowledge', 2), ('engineering', 2),...,('planning', 1), ('mapping', 1)]) ##篇幅原因，中间省略
```

对词频进行倒序排序：

```python
ai_result=sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
```

`ai_result`的输出结果为：

```python
[('', 17),
 ('of', 11),
 ('to', 11),
 ('and', 10),
 ('a', 7),
 ('the', 7),
 ... ##篇幅原因，中间省略
 ('navigation', 1),
 ('along', 1),
 ('localization', 1),
 ('motion', 1),
 ('planning', 1),
 ('mapping', 1)]
```

我们发现词频很高的词基本都是没有意义的**停用词**，因此我们需要去掉这些停用词。首先，先加载停用词：

```python
##加载停用词
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords') ##需先下载stopwords资源
stop_words=stopwords.words('English') ##英文停用词
```
>from...import...和import...
>
>import moudle要调用moudle中的Function\_a()必需完整的写为：moudle.Function_a()
>
>from moudle import * 的话，直接调用Function\_a()就可以了。比如：from sys import argv，可以直接argv，而不用sys.argv

去除停用词，得到最后结果：

```python
for k,v in ai_result:
    if k not in stop_words:
        if k != '': ##把空字符去除
            print(k,v)
```

结果见下：

```python
learning 5
inputs 4
ai 3
machine 3
object 3
...##篇幅原因，中间省略
navigation 1
along 1
localization 1
motion 1
planning 1
mapping 1
```

⚠️在有列表或字典的循环结构中，若对列表或字典进行改、增、删等操作，会使得在迭代过程中，列表元素的值或者位置（集体前移或者后移）发生变化，或者是字典长度发生变化等，此时程序会报错，所以应该尽量避免。

# 2.词频统计（方法二）

另一种统计词频的方法，更为方便快捷：

```python
from collections import Counter
c=Counter(ai_text_split)
```
输出为：

```python
Counter({'knowledge': 2,
         'engineering': 2,
         'is': 6,
         'a': 7,
         'core': 2,
         'part': 2,
         ... ##篇幅原因，中间省略
         'navigation': 1,
         'along': 1,
         'localization': 1,
         'motion': 1,
         'planning': 1,
         'mapping': 1})
```
(输出为字典形式)

删除停用词：

```python
for sw in stop_words:
	del c[sw]
del c[''] ##删除空字符
```

输出词频最高的3个词：

```python
c.most_common(3) ##输出为：[('learning', 5), ('inputs', 4), ('ai', 3)]
```

# 3.python中module和package的区别

* Module：python文件，后缀为xxx.py（单个Module）
* Package：一组相关的Module组合而成（多个Module）

在Python中，一个.py文件就称之为一个模块（Module），当一个模块编写完毕，就可以被其他地方引用。使用模块还可以避免函数名和变量名冲突，相同名字的函数和变量完全可以分别存在不同的模块中（尽量不要和python内置函数名字冲突）。为了避免模块名冲突，Python又引入了按目录来组织模块的方法，称为包（Package）。

![](https://ws1.sinaimg.cn/large/006tNbRwly1fyqzaaxattj30cq0cwt9h.jpg)

上图是一个多级层次的包结构,共有两个包目录：mycompany和web。⚠️每个包目录下面都会有一个`__init__.py`的文件，这个文件是必须存在的，否则，Python就把这个目录当成普通目录，而不是一个包。

两个文件`utils.py`的模块名分别是`mycompany.utils`和`mycompany.web.utils`。

# 4.参考资料
1.[模块（廖雪峰的官方网站）](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318447437605e90206e261744c08630a836851f5183000)
