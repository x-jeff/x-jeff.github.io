---
layout:     post
title:      【Tensorflow基础】第十一课：TFRecord的生成
subtitle:   TFRecord，tf.Graph().as_default()，tf.python_io.TFRecordWriter，tf.train.BytesList，tf.train.Int64List，tf.train.FloatList，tf.train.Feature，tf.train.Features，tf.train.Example，SerializeToString
date:       2022-02-09
author:     x-jeff
header-img: blogimg/20220209.jpg
catalog: true
tags:
    - Tensorflow Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.tfrecord

## 1.1.什么是tfrecord

tfrecord是Google官方推荐的一种数据格式，是Google专门为TensorFlow设计的一种数据格式。实际上，tfrecord是一种二进制文件，其能更好的利用内存，其内部包含了多个`tf.train.Example`，而`Example`是protocol buffer(protobuf)数据标准的实现。在一个`Example`中包含了一系列的`tf.train.feature`属性，而每一个feature是一个key-value的键值对。其中，key是string类型，而value的取值有三种：

* `bytes_list`：可以存储string和byte两种数据类型。
* `float_list`：可以存储float(float32)与double(float64)两种数据类型。
* `int64_list`：可以存储：bool，enum，int32，uint32，int64，uint64。

tfrecord并非是TensorFlow唯一支持的数据格式，也可以使用CSV或文本等格式，但是对于TensorFlow来说，tfrecord是最友好的，也是最方便的。

Google官方推荐对于中大数据集，先将数据集转化为tfrecord数据（`.tfrecords`），这样可加快在数据读取，预处理中的速度。

## 1.2.生成tfrecord

代码中用到的API：

👉`tf.Graph().as_default()`：获取当前默认的计算图。

👉`tf.python_io.TFRecordWriter`：创建一个TFRecordWriter对象。

### 1.2.1.`tf.train.Example`

比如有txt文件，我们按行将其读入inputs：

```
inputs[0] : 21
inputs[1] : This is a test data file.
inputs[2] : We will convert this text file to bin file.
```

原始数据可以用`tf.train.BytesList`（处理非数值数据）、`tf.train.Int64List`（处理整型数据）、`tf.train.FloatList`（处理浮点型数据）来处理。

```python
data_id = tf.train.Int64List(value=[int(inputs[0])])
data = tf.train.BytesList(value=[bytes(' '.join(inputs[1:]), encoding='utf-8')])
```

设置`tf.train.Feature`：

```python
tf.train.Feature(int64_list=data_id),
tf.train.Feature(bytes_list=data)
```

将多个`tf.train.Feature`以字典的形式传给`tf.train.Features`：

```python
feature_dict = {
    "data_id": tf.train.Feature(int64_list=data_id),
    "data": tf.train.Feature(bytes_list=data)
}
features = tf.train.Features(feature=feature_dict)
```

建立`Example`：

```python
example = tf.train.Example(features=features)
```

序列化`Example`为字符串：

```python
example_str = example.SerializeToString()
```

序列化后的example_str便可直接写入tfrecord。

# 2.代码地址

1. [TFRecord的生成](https://github.com/x-jeff/Tensorflow_Code_Demo/tree/master/Demo10)

# 3.参考资料

1. [TFRecord - TensorFlow 官方推荐的数据格式](https://zhuanlan.zhihu.com/p/50808597)
2. [tf.train.Example的用法](https://blog.csdn.net/hfutdog/article/details/86244944)