---
layout:     post
title:      【Tensorflow基础】第十四课：CNN在自然语言处理的应用
subtitle:   tf.app.flags，tf.app.run，tf.flags，re.sub，VocabularyProcessor，np.random.permutation，tf.ConfigProto，compute_gradients，apply_gradients，tf.nn.zero_fraction，os.path.abspath，os.path.curdir，datetime.datetime.now().isoformat()，yield，tf.train.global_step
date:       2022-05-02
author:     x-jeff
header-img: blogimg/20220502.jpg
catalog: true
tags:
    - Tensorflow Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.CNN在自然语言处理的应用

CNN通常应用于计算机视觉领域。但近几年CNN也开始应用于自然语言处理，并取得了一些引人注目的成绩。

CNN应用于NLP任务，处理的往往是以矩阵形式表达的句子或文本。矩阵中的每一行对应于一个分词元素，一般是一个单词，也可以是一个字符。假设我们一共有10个词，每个词都用128维的向量表示，那么我们就可以得到一个$10 \times 128$维的矩阵。比如：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/TensorflowSeries/Lesson14/14x1.png)

# 2.代码实现

代码基于[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)稍作修改。任务描述：对电影评论进行二分类（好评或者差评）。

👉导入必要的包：

```python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
```

👉定义一些模型参数：

```python
# Data loading params
## 验证集占比
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
## 正样本路径
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
## 负样本路径                    
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")
                       
# Model Hyperparameters
## 词向量长度
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
## 卷积核大小
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
## 每一种卷积核的个数
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
## dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
## L2正则化参数
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
                       
# Training parameters
## batch size
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
## epoch数
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
## 每多少step测试一次
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
## 每多少step保存一次模型
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
## 最多保存多少个模型
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
## tensorflow会自动选择一个存在并且支持的设备来运行operation
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
## 获取你的operations和tensor被指派到哪个设备上运行
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices") 

# flags解析
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

# 打印所有参数
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")               
```

```
Parameters:
ALLOW_SOFT_PLACEMENT=True
ALSOLOGTOSTDERR=False
BATCH_SIZE=64
CHECKPOINT_EVERY=100
DEV_SAMPLE_PERCENTAGE=0.1
DROPOUT_KEEP_PROB=0.5
EMBEDDING_DIM=128
EVALUATE_EVERY=100
FILTER_SIZES=3,4,5
L2_REG_LAMBDA=0.0
LOG_DEVICE_PLACEMENT=False
LOG_DIR=
LOGTOSTDERR=False
NEGATIVE_DATA_FILE=./data/rt-polaritydata/rt-polarity.neg
NUM_CHECKPOINTS=5
NUM_EPOCHS=200
NUM_FILTERS=128
ONLY_CHECK_ARGS=False
OP_CONVERSION_FALLBACK_TO_WHILE_LOOP=False
PDB_POST_MORTEM=False
POSITIVE_DATA_FILE=./data/rt-polaritydata/rt-polarity.pos
PROFILE_FILE=None
RUN_WITH_PDB=False
RUN_WITH_PROFILING=False
SHOWPREFIXFORINFO=True
STDERRTHRESHOLD=fatal
TEST_RANDOM_SEED=301
TEST_RANDOMIZE_ORDERING_SEED=
TEST_SRCDIR=
TEST_TMPDIR=/var/folders/qg/0r2bywpn6s16dsr8j9xyjnm80000gn/T/absl_testing
USE_CPROFILE_FOR_PROFILING=True
V=-1
VERBOSITY=-1
XML_OUTPUT_FILE=
```

`tf.app.flags`主要用于处理命令行参数的解析工作，支持接受命令行传递参数。跟它配合的还有一个`tf.app.run`函数，用于执行程序中main函数并解析命令行参数。比如：

```python
##test_use.py
import tensorflow as tf

##第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_string('str_name', 'def_v_1', "descrip1")
tf.app.flags.DEFINE_integer('int_name', 10, "descript2")
tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")
FLAGS = tf.app.flags.FLAGS


##必须带参数，否则：'TypeError: main() takes no arguments (1 given)';   ##main的参数名随意定义，无要求
def main(_):
    print(FLAGS.str_name)
    print(FLAGS.int_name)
    print(FLAGS.bool_name)


if __name__ == '__main__':
    tf.app.run()  # tf.app.run()的作用：先处理flag解析，然后执行main函数，
```

输出为：

```
def_v_1
10
False
```

可以通过命令行修改默认值，比如：

```
$ python test_use.py --str_name="def_v_2"
```

运行结果为：

```
def_v_2
10
False
```

在老版本1.0+的tensorflow中使用`tf.app.flags`来定义参数，新版本2.0+用`tf.flags`来定义参数。

👉读入数据集：

```python
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # 不是特定字符都变成空格
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # 加空格
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    # 匹配2个或多个空白字符变成一个" "空格
    string = re.sub(r"\s{2,}", " ", string)
    # 去掉句子首尾的空白符，再转小写
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding="utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding="utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
```

`re.sub`用于替换字符串中的匹配项：

```python
re.sub(pattern, repl, string, count=0, flags=0)
```

参数详解（前三个为必选参数，后两个为可选参数）：

* `pattern`：正则中的模式字符串。
* `repl`：替换的字符串，也可为一个函数。
* `string`：要被查找替换的原始字符串。
* `count`：模式匹配后替换的最大次数，默认0表示替换所有的匹配。
* `flags`：编译时用的匹配模式，数字形式。

```python
#!/usr/bin/python3
import re

phone = "2004-959-559 # 这是一个电话号码"

# 删除注释
num = re.sub(r'#.*$', "", phone)
print("电话号码 : ", num)

# 移除非数字的内容
num = re.sub(r'\D', "", phone)
print("电话号码 : ", num)
```

```
电话号码 :  2004-959-559 
电话号码 :  2004959559
```

👉建立字典：

```python
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
```

Tensorflow提供了`VocabularyProcessor`函数用于构建词典，得到的数组`x`中的每一行对应一个句子，数字对应单词在词典中的索引，`x`的列数通常设为最长句子的单词数，单词数不足的句子用0补齐：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/TensorflowSeries/Lesson14/14x2.png)

👉将数据打乱：

```python
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
```

`np.random.permutation`用于随机排序：

```
>>> np.random.permutation(10)
array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random
    
>>> np.random.permutation([1, 4, 9, 12, 15])
array([15,  1,  9,  4, 12]) # random
    
>>> arr = np.arange(9).reshape((3, 3))
>>> np.random.permutation(arr)
array([[6, 7, 8], # random
       [0, 1, 2],
       [3, 4, 5]])
```

👉划分训练集和测试集：

```python
# Split train/test set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
```

👉传入参数：

```python
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
```

`tf.ConfigProto`在创建会话的时候进行参数配置，比如GPU、CPU、显存等。`TextCNN`定义了第1部分中所示的网络结构，详见博文末尾所附的代码链接中的`text_cnn.py`，很简单的实现，这里不再赘述。

👉定义训练：

```python
# Define Training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
```

通常所用的`minimize()`内部其实也是分两部分：第一步，`compute_gradients`根据loss目标函数计算梯度；第二步，`apply_gradients`使用计算得到的梯度来更新对应的Variable。之所以分开，是因为有时候需要对计算出来的梯度做一定的修正，以防[梯度爆炸或梯度消失](http://shichaoxin.com/2020/02/07/深度学习基础-第十三课-梯度消失和梯度爆炸/)。

👉将梯度的变化记录到tensorboard中：

```python
# Keep track of gradient values and sparsity (optional)
grad_summaries = []
# g : gradient
# v : variable
for g, v in grads_and_vars:
    if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)
```

>TensorBoard的使用：[【Tensorflow基础】第六课：TensorBoard的使用](http://shichaoxin.com/2020/07/29/Tensorflow基础-第六课-TensorBoard的使用/)。

`tf.nn.zero_fraction`的作用是将输入的tensor中0元素在所有元素中所占的比例计算并返回，即返回输入tensor的0元素的个数与输入tensor的所有元素的个数的比值。

👉定义输出路径：

```python
# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
```

`os.path.abspath`用于获取指定文件或目录的绝对路径。`os.path.curdir`返回'.'，表示当前路径。

👉添加更多信息到summary：

```python
# Summaries for loss and accuracy
loss_summary = tf.summary.scalar("loss", cnn.loss)
acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# Dev summaries
dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
```

👉模型保存和我们构建的字典：

```python
# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

# Write vocabulary
vocab_processor.save(os.path.join(out_dir, "vocab"))
```

👉初始化Variable：

```python
# Initialize all variables
sess.run(tf.global_variables_initializer())
```

👉定义训练和测试步骤：

```python
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)

def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    if writer:
        writer.add_summary(summaries, step)
```

`datetime.datetime.now().isoformat()`：

```python
import datetime
datetime.datetime.now() #datetime.datetime(2022, 5, 24, 21, 13, 0, 907223)
datetime.datetime.now().isoformat() #返回字符串：'2022-05-24T21:13:11.881698'
```

👉产生batch：

```python
batches = data_helpers.batch_iter(
    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
```

`data_helpers.batch_iter`的定义如下：

```python
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    print("num_batches_per_epoch:",num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
```

`yield`的用法见本文第3部分。

👉训练部分代码：

```python
# Training loop. For each batch...
for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
    if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
```

`tf.train.global_step`（相当于batch）每执行一次，`global_step`就会加1。

# 3.`yield`

首先介绍一下**生成器（generator）**，其提供一种可以边循环边计算的机制。生成器是解决使用序列存储大量数据时，内存消耗大的问题。我们可以根据存储数据的某些规律，演算为算法，在循环过程中通过计算得到，这样可以不用创建完整序列，从而大大节省占用空间。`yield`是实现生成器方法之一，当函数使用`yield`方法，则该函数就成为了一个生成器。调用该函数，就等于创建了一个生成器对象。接下来通过几个例子来进一步了解`yield`。

```python
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:",res)
g = foo()
print(next(g))
print("*"*20)
print(next(g))
```

输出为：

```
starting...
4
********************
res: None
4
```

代码执行顺序解释：

1. 程序开始执行以后，因为`foo`函数中有`yield`关键字，所以`foo`函数并不会真的执行，而是先得到一个生成器`g`（相当于一个对象）。
2. 直到调用`next`方法，`foo`函数正式开始执行，先执行`foo`函数中的`print`方法，然后进入`while`循环。
3. 程序遇到`yield`关键字，然后把`yield`想象成return，return了一个4之后，程序停止，并没有执行赋值给`res`操作，此时`next(g)`语句执行完成，所以输出的前两行是执行`print(next(g))`的结果。
4. 程序执行`print("*"*20)`。
5. 开始执行下面的`print(next(g))`，这个时候和上面那个差不多，不过不同的是，这个时候是从刚才那个`next`程序停止的地方开始执行的，也就是要执行`res`的赋值操作，这时候要注意，这个时候赋值操作的右边是没有值的（因为刚才那个是return出去了，并没有给赋值操作的左边传参数），所以这个时候`res`赋值是None，所以接着下面的输出就是`res: None`。
6. 程序会继续在`while`里执行，又一次碰到`yield`，这个时候同样return出4，然后程序停止，`print`函数输出的4就是这次return出的4。

再看另外一个例子：

```python
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:",res)
g = foo()
print(next(g))
print("*"*20)
print(g.send(7))
```

输出为：

```
starting...
4
********************
res: 7
4
```

前4步和上一个例子是一样的。第5步：程序执行`g.send(7)`，程序会从`yield`关键字那一行继续向下运行，`send`会把7这个值赋给`res`变量。第6步：由于`send`方法中包含`next()`方法，所以程序会继续向下运行`print`，然后再次进入`while`循环。第7步：程序执行再次遇到`yield`关键字，`yield`会返回后面的值，然后程序再次暂停，直到再次调用`next`方法或`send`方法。

最后通过一个例子解释下使用生成器的一个原因。例如：

```python
for n in range(1000):
	print(n)
```

此时，`range(1000)`默认生成一个含有1000个数的list，所以会很占内存。此时可以使用`yield`：

```python
def foo(num):
    print("starting...")
    while num<1000:
        num=num+1
        yield num
for n in foo(0):
    print(n)
```

此时，`foo(0)`会一个数一个数的返回，节省了内存（个人感觉有点像C++中的[static](http://shichaoxin.com/2021/12/04/C++基础-第三十四课-函数基础/#22局部静态对象)）。

# 4.代码地址

1. [CNN在自然语言处理的应用](https://github.com/x-jeff/Tensorflow_Code_Demo/tree/master/Demo13)

# 5.参考资料

1. [tf.app.flags()和tf.flags()的用法及区别](https://blog.csdn.net/pearl8899/article/details/108061781)
2. [Tensorflow使用flags定义命令行参数详解](https://blog.csdn.net/qq_36653505/article/details/81124533)
3. [Python3 正则表达式（菜鸟教程）](https://www.runoob.com/python3/python3-reg-expressions.html#flags)
4. [以终为始：compute\_gradients 和 apply\_gradients](https://zhuanlan.zhihu.com/p/343628982)
5. [Tensorflow-tf.nn.zero_fraction()详解](https://blog.csdn.net/fegang2002/article/details/83539768)
6. [python中yield的用法详解——最简单，最清晰的解释](https://blog.csdn.net/mieleizhi0522/article/details/82142856/)
7. [Python 生成器yield](https://baijiahao.baidu.com/s?id=1725787762423255527&wfr=spider&for=pc)