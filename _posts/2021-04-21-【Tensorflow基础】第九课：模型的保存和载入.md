---
layout:     post
title:      【Tensorflow基础】第九课：模型的保存和载入
subtitle:   tf.train.Saver()，saver.save()，saver.restore()，ckpt模型
date:       2021-04-21
author:     x-jeff
header-img: blogimg/20210421.jpg
catalog: true
tags:
    - Tensorflow Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.`tf.train.Saver()`

`tf.train.Saver()`用于保存和加载模型。

```python
saver=tf.train.Saver()
```

`tf.train.Saver()`参数见下：

```python
def __init__(
	self,
	var_list=None,
	reshape=False,
	sharded=False,
	max_to_keep=5,
	keep_checkpoint_every_n_hours=10000.0,
	name=None,
	restore_sequentially=False,
	saver_def=None,
	builder=None,
	defer_build=False,
	allow_empty=False,
	write_version=saver_pb2.SaverDef.V2,
	pad_step_number=False,
	save_relative_paths=False,
	filename=None)
```

部分参数解释：

1. `var_list`：指定要保存和恢复的变量。
2. `max_to_keep`：是经常会用到的一个参数。用于设置保存模型的个数（默认为`max_to_keep=5`，即保存最近的5个模型）。若`max_to_keep`设置为None或0，则保存所有的模型。
3. `keep_checkpoint_every_n_hours`：每n个小时保存一次模型。

## 1.1.`saver.save()`

```python
def save(
	self,
	sess,
	save_path,
	global_step=None,
	latest_filename=None,
	meta_graph_suffix="meta",
	write_meta_graph=True,
	write_state=True,
	strip_default_attrs=False,
	save_debug_info=False)
```

部分参数解释：

1. `sess`：Session。
2. `save_path`：模型保存路径。例如：`saver.save(sess, 'net/my_net.ckpt')`。
3. `global_step`：用来给模型文件名添加数字标记。例如：`saver.save(sess, 'my-model', global_step=0)`，保存得到的模型文件名为：`'my-model-0'`。

## 1.2.`saver.restore()`

```python
def restore(self, sess, save_path)
```

参数解释：

1. `sess`：Session。
2. `save_path`：模型路径。例如：`saver.restore(sess, 'net/my_net.ckpt')`。

导入模型之前，必须重新再定义一遍变量。但是并不需要全部变量都重新进行定义，只定义我们需要的变量就行了。

可以使用`tf.train.latest_checkpoint()`来自动获取最后一次保存的模型。如：

```python
model_file=tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)
saver.restore(sess,model_file)
```

# 2.ckpt模型

使用`saver.save()`将模型保存为ckpt格式，会生成以下四个文件：

![](https://github.com/x-jeff/BlogImage/raw/master/TensorflowSeries/Lesson9/9x1.png)

1. `my_net.ckpt.meta`：保存了Tensorflow计算图的结构，即网络结构。
2. `my_net.ckpt.index`和`my_net.ckpt.data-00000-of-00001`：保存了所有变量的取值。
3. `checkpoint`：保存了一个目录下所有的模型文件列表。

# 3.代码地址

1. [模型的保存和载入](https://github.com/x-jeff/Tensorflow_Code_Demo/tree/master/Demo8)