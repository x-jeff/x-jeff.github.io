---
layout:     post
title:      【Tensorflow基础】第十课：Inception-v3的训练和检测
subtitle:   Inception-v3，os.walk，tf.gfile.FastGFile，get_tensor_by_name
date:       2022-01-22
author:     x-jeff
header-img: blogimg/20220122.jpg
catalog: true
tags:
    - Tensorflow Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.下载Inception-v3并查看其结构

>Inception-v3详细介绍请见：[【论文阅读】Rethinking the Inception Architecture for Computer Vision](http://shichaoxin.com/2021/11/29/论文阅读-Rethinking-the-Inception-Architecture-for-Computer-Vision/)。

核心部分的代码：

```python
with tf.Session() as sess:
    # 创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
```

👉`tf.gfile.FastGFile`用于实现对图片的读取。第一个参数为图片所在路径。第二个参数为图片的解码方式：‘r’表示UTF-8编码；‘rb’表示非UTF-8编码。

通过[TensorBoard](http://shichaoxin.com/2020/07/29/Tensorflow基础-第六课-TensorBoard的使用/)可以可视化得到的模型。

>好久不用tensorboard，这次使用突然报错：ValueError: Duplicate plugins for name projector。解决办法：在所用的conda虚拟环境下，删除tensorboard-1.14.0.dist-info类似命名的文件夹。

# 2.使用Inception-v3做各种图像的识别

使用第1部分下载好的模型进行图像识别（共1000个类别）。这1000个类别的信息放在“imagenet\_2012\_challenge\_label\_map\_proto.pbtxt”和“imagenet\_synset\_to\_human\_label\_map.txt”中。“imagenet\_2012\_challenge\_label\_map\_proto.pbtxt”中的数据格式见下：

```
entry {
  target_class: 449
  target_class_string: "n01440764"
}
```

449为类别编号，n01440764在“imagenet\_synset\_to\_human\_label\_map.txt”中可找到该类别对应的字符串描述：

```
n01440764	tench, Tinca tinca
```

核心部分代码见下：

```python
# 创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    # 遍历目录
    for root, dirs, files in os.walk('images/'):
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
            predictions = np.squeeze(predictions)  # 把结果转为1维数据
```

👉`get_tensor_by_name`：所有的tensor都有string格式的名字，可以通过名字来fetch tensor。

👉`os.walk`可用于遍历一个目录，返回的是一个三元组：

1. root：当前正在遍历的这个文件夹的本身的地址。
2. dirs：是一个list，内容是该文件夹中所有的目录的名字（不包括子目录）。
3. files：同样是list，内容是该文件夹中所有的文件（不包括子目录）。

![](https://github.com/x-jeff/BlogImage/raw/master/TensorflowSeries/Lesson10/10x1.jpg)

例如上图的分类结果（列出了概率最高的5个类别）为：

```
images/car.jpg
sports car, sport car (score = 0.89100)
grille, radiator grille (score = 0.02280)
car wheel (score = 0.02095)
crash helmet (score = 0.00919)
convertible (score = 0.00335)
```

# 3.训练自己的Inception-v3模型

修改Inception-v3的输出层（即最后一个pooling层后面的结构），并使用自己的数据只训练我们修改的部分（最后一个pooling层及其之前层的结构和参数不变）。

>可以到[http://www.robots.ox.ac.uk/~vgg/data/](http://www.robots.ox.ac.uk/~vgg/data/)下载想要的数据集。

通过TensorFlow官方提供的[retrain.py](https://github.com/tensorflow/hub/tree/master/examples/image_retraining)来快速实现重训练。

>如果想使用自己的数据从头训练一个模型，可以参考：[链接](https://github.com/tensorflow/models/tree/master/research/slim)。

# 4.代码地址

1. [Inception-v3的训练和检测](https://github.com/x-jeff/Tensorflow_Code_Demo/tree/master/Demo9)

# 5.参考资料

1. [tf.gfile.FastGFile()](https://blog.csdn.net/william_hehe/article/details/78821715)
2. [Python os.walk() 方法](https://www.runoob.com/python/os-walk.html)