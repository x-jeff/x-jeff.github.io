---
layout:     post
title:      【Tensorflow基础】第七课：卷积神经网络的实现
subtitle:   tf.nn.conv2d()，padding详解，tf.nn.max_pool()
date:       2020-09-11
author:     x-jeff
header-img: blogimg/20200911.jpg
catalog: true
tags:
    - Tensorflow Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.卷积神经网络

卷积神经网络的相关介绍请戳👉：[【深度学习基础】第二十八课：卷积神经网络基础](http://shichaoxin.com/2020/07/04/深度学习基础-第二十八课-卷积神经网络基础/)。

# 2.使用tensorflow实现CNN

先介绍可能会用到的API。

## 2.1.`tf.nn.conv2d()`

`tf.nn.conv2d()`用于构建网络的卷积层，这里的`2d`指的是二维卷积核，也是最为常用的。API详细参数见下：

```python
def conv2d(  
    input,
    filter=None,
    strides=None,
    padding=None,
    use_cudnn_on_gpu=True,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None,
    filters=None)
```

部分常用参数解释：

1. `input`：输入`[input_batch_size,input_height,input_width,input_channel]`。
2. `filter`：卷积核`[filter_height,filter_width,filter_channel,filter_number]`。通常情况下，`filter_channel=input_channel`。
3. `stride`：步长`[stride_batch_size, stride_height, stride_width, stride_channel]`，分别指在`input`四个维度上的步长。通常情况下，`stride_batch_size = stride_channel =1`且`stride_height=stride_width`。
4. `padding`提供`SAME`和`VALID`两种池化方式。

### 2.1.1.`padding`

这里额外多说一点关于`padding`参数的注意事项。

参数`padding`除了用字符串`SAME`或者`VALID`指明其方式外，还可以用具体的数值设置其具体补充的行数（或列数）。例如在`data_format="NHWC"`格式下，`padding`可以通过如下方式赋值：

```python
padding =[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
```

即`NHWC`的每个维度两边都需要进行padding。而维度`N`和`C`，通常不进行padding。

然后我们再来说说`SAME`和`VALID`有什么区别。在[【深度学习基础】第二十八课：卷积神经网络基础](http://shichaoxin.com/2020/07/04/深度学习基础-第二十八课-卷积神经网络基础/#2padding)一文中，我们初步了解了这两种方式的作用机制。那么今天我们来进一步分析下，`SAME`和`VALID`输出结果的维度该怎么确定。

`SAME`输出的维度：

$$o=\lceil \frac{i}{s} \rceil$$

`VALID`输出的维度：

$$o=\lfloor \frac{i-k}{s} + 1 \rfloor$$

上述式子中，$o$为输出的height（或width），$i$为输入的height（或width），$s$为步长，$k$为卷积核的大小（假设`filter_height=filter_width=k`）。

例如有一个$2\times 3$的平面，用$2\times 2$并且步长为2的窗口对其进行`pooling`操作：

* 使用`SAME`的padding方式，得到$1\times 2$的平面。
* 使用`VALID`的padding方式，得到$1\times 1$的平面。

## 2.2.`tf.nn.max_pool()`

`tf.nn.max_pool()`用来构建网络的池化层（max pooling）。API详细参数见下：

```python
def max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format="NHWC",
    name=None,
    input=None)
```

其中，参数`value`是一个四维的输入，`ksize`为用于池化操作的核的维度，通常为`[1,height,width,1]`。其他参数和`tf.nn.conv2d()`中的一样。

# 3.代码地址

1. [使用tensorflow实现CNN](https://github.com/x-jeff/Tensorflow_Code_Demo/tree/master/Demo6)

# 4.参考资料

1. [tf中的padding方式SAME和VALID有什么区别?](https://bugxch.github.io/post/tf中的padding方式same和valid有什么区别/)