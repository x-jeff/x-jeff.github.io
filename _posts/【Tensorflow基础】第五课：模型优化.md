# 1.前言

我们以[【Tensorflow基础】第四课：手写数字识别](http://shichaoxin.com/2020/03/26/Tensorflow基础-第四课-手写数字识别/)中构建的手写数字识别模型为例，看一下深度学习中常用的模型优化算法怎么用tensorflow实现。

# 2.修改代价函数

在之前的模型中，我们用的是均方误差作为cost function。现在我们使用更合适的[交叉熵损失函数](http://shichaoxin.com/2019/09/04/深度学习基础-第二课-softmax分类器和交叉熵损失函数/)作为cost function：

```python
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
```

结果对比见下：

![](https://github.com/x-jeff/BlogImage/raw/master/TensorflowSeries/Lesson5/5x1.png)

很明显，交叉熵损失函数效果更好，收敛速度更快。

# 3.修改网络

在第2部分的基础上，对网络的结构和参数进行了如下修改：

1. 添加隐藏层。
2. 随机初始化权重。
3. dropout防止过拟合。

>为什么要随机初始化权重？请戳👉[【深度学习基础】第十三课：梯度消失和梯度爆炸](http://shichaoxin.com/2020/02/07/深度学习基础-第十三课-梯度消失和梯度爆炸/)。
>
>关于dropout的详细介绍，请戳👉[【深度学习基础】第十一课：正则化](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/)。

代码如下：

```python
#第一层
W1=tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1=tf.Variable(tf.zeros([2000])+0.1)
A1=tf.nn.tanh(tf.matmul(x,W1)+b1)
A1_drop=tf.nn.dropout(A1,keep_prob)
#第二层
W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2=tf.Variable(tf.zeros([2000])+0.1)
A2=tf.nn.tanh(tf.matmul(A1_drop,W2)+b2)
A2_drop=tf.nn.dropout(A2,keep_prob)
#第三层
W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3=tf.Variable(tf.zeros([1000])+0.1)
A3=tf.nn.tanh(tf.matmul(A2_drop,W3)+b3)
A3_drop=tf.nn.dropout(A3,keep_prob)
#输出层
W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(A3_drop,W4)+b4)
```

函数`tf.truncated_normal`：

```python
truncated_normal(
	shape,#输出张量围度
	mean=0.0,#均值
	stddev=1.0,#标准差
	dtype=dtypes.float32,#输出类型
	seed=None,#随机数种子
	name=None#运算名称
)
```

该函数可产生截断正态分布随机数，取值范围为$[mean-2\times stddev,mean+2\times stddev]$。

`keep_prob`在训练时设为0.7，预测时不能使用dropout，因此预测时`keep_drop`设为1.0。

修改后预测结果为：

![](https://github.com/x-jeff/BlogImage/raw/master/TensorflowSeries/Lesson5/5x2.png)

相比第2部分，结果又有了进一步的提升。