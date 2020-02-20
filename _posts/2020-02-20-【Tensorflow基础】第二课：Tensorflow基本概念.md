---
layout:     post
title:      【Tensorflow基础】第二课：Tensorflow基本概念
subtitle:   Graph，Session，Tensor，Operation，Feed，Fetch
date:       2020-02-20
author:     x-jeff
header-img: blogimg/20200220.jpg
catalog: true
tags:
    - Tensorflow Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Tensorflow基本概念

## 1.1.Graph

**Graph（图）：**表示计算任务，用于搭建神经网络的计算任务。

## 1.2.Session

**Session（会话）：**在Session中执行Graph。

## 1.3.Tensor

**Tensor（张量）：**张量就是一种拥有不同维度的数据结构。Tensor是Tensorflow中的基本数据结构：

* **0阶张量（标量）：**一个数。
* **1阶张量（向量）：**一维数组。
* **2阶张量（矩阵）：**二维数据。
* ......（以此类推）

⚠️Tensorflow中的一切数据都属于Tensor类型。

⚠️任何Tensor数据在运算之前都是得不到具体数值的。

神经网络中通常需要以下几种数据类型：

1. 可更新的参数：包括权重（weights），偏置项（bias）。这些参数将在训练过程中不断更新。
	* 对应数据类型：`tf.Variable`。
2. 独立于模型存在的数据：数据集中的数据需要“喂给”网络，包括输入数据、输出端的groundtruth。
	* 对应数据类型：`tf.placeholder`。
3. 常量。
	* 对应数据类型：`tf.constant`。

## 1.4.Operation	

**Operation（操作）：**图中的节点称之为op(operation)。但凡是op，都需要通过Session运行之后，才能得到结果。

⚠️Graph是由一系列op构成的。

## 1.5.Feed、Fetch

使用Feed和Fetch可以为任意的操作赋值或者从其中获取数据。

## 1.6.例子

![](https://github.com/x-jeff/BlogImage/raw/master/TensorflowSeries/Lesson2/2x1.jpg)

如上图中的例子，一个Session中有两个Graph。

接下来我们通过一段简单的代码来进一步的了解。首先定义两个constant类型的op：

```python
import tensorflow as tf
tensor1 = tf.constant([[3,4]]) 
tensor2 = tf.constant([[5],[6]])
```

`tensor1`是一个$1\times 2$维的向量，`tensor2`是一个$2\times 1$维的向量。现在新建一个op，计算两个tensor的点积：

```python
tensor3 = tf.matmul(tensor1,tensor2)
print(tensor3)
```

❗️`tf.matmul()`可用于计算点积。

上述代码输出为：`Tensor("MatMul:0", shape=(1, 1), dtype=int32)`。可见得到的`tensor3`并不是一个具体的数值，而是一个Tensor。这是因为在运算开始之前Tensor是得不到具体数值的，而运算流程（也就是Graph）必须在Session中运行：

```python
#定义一个新的Session
sess = tf.Session()
#开始计算Graph
result = sess.run(tensor3)
print(result)
#关闭Session
sess.close()
```

可得到`result`为`[[39]]`。

另外，可简化省去`sess.close()`：

```python
with tf.Session() as sess:
    result = sess.run(tensor3)
    print(result)
```

得到的`result`的结果都是一样的。

# 2.Variable

这一部分我们着重介绍tensorflow中非常常用的一种tensor：Variable。

```python
import tensorflow as tf
v1 = tf.Variable([1,2])
c1 = tf.constant([3,4])
sub = tf.subtract(v1,c1) #定义一个减法op
add = tf.add(v1,c1) #定义一个加法op

init = tf.global_variables_initializer() #初始化全局变量

with tf.Session() as sess:
    sess.run(init)
    sub_result = sess.run(sub)
    add_result = sess.run(add)
    print(sub_result) #[-2,-2]
    print(add_result) #[4,6]
```

‼️Variable必须初始化：

* `init = tf.global_variables_initializer()`可以初始化全局所有变量。
* `sess.run(v1.initializer)`只初始化变量`v1`。

再举另外一个例子：

```python
v2 = tf.Variable(0,name="counter")
new_v2 = tf.add(v2,1)
update_v2 = tf.assign(v2,new_v2) #用于赋值操作
init = tf.global_variables_initializer() #初始化全局变量
with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(update_v2)
        print(sess.run(v2))
```

输出为：

```
1
2
3
4
5
```

# 3.Feed、Fetch

## 3.1.Fetch

`Fetch`字面意思是“取来”，即得到运行结果。在`sess.run()`中，即可fetch一个op的值，也可同时fetch多个op的值。

```python
c1=tf.constant(1)
c2=tf.constant(2)
c3=tf.constant(3)
add1=tf.add(c2,c3)
mul1=tf.multiply(c1,add1)
with tf.Session() as sess:
    result1=sess.run([mul1,add1])
    print(result1) #[5,5]
```

⚠️`sess.run([mul1,add1])`是以列表`list`的形式，因此不要忘了`[]`。

## 3.2.Feed

`Feed`字面意思是“喂养”，即喂入数据。

```python
p1=tf.placeholder(tf.float32)
p2=tf.placeholder(tf.float32)
mul2=tf.multiply(p1,p2)
with tf.Session() as sess:
    result2=sess.run(mul2,feed_dict={p1:[5.],p2:[.3]})
    print(result2)
```

一开始定义的`p1`、`p2`只是两个占位符（32位float类型），并没有定义实际的值。因此在sess.run()的时候需要通过`Feed`操作传入数据。

⚠️`Feed`操作是通过python字典的形式传入数据的。

* `sess.run(mul2,feed_dict={p1:[5.],p2:[.3]})`输出为一个numpy数组`ndarray`：[1.5]。
* `sess.run(mul2,feed_dict={p1:5.,p2:.3})`输出为一个数值`float`：1.5。

# 4.代码地址

1. [Tensorflow基本概念及应用实例](https://github.com/x-jeff/Tensorflow_Code_Demo/tree/master/Demo1)

# 5.参考资料

1. [Tensorflow入门教程（1）(作者：Seventeen)](https://zhuanlan.zhihu.com/p/34530755)