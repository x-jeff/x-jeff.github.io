---
layout:     post
title:      【论文阅读】ImageNet Classification with Deep Convolutional Neural Networks
subtitle:   ILSVRC2012(分类任务冠军/定位任务冠军)：AlexNet
date:       2021-02-03
author:     x-jeff
header-img: blogimg/20210203.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.背景（Introduction）

AlexNet在ILSVRC-2010和ILSVRC-2012中都取得了前几名的好成绩（ILSVRC-2012冠军）。

AlexNet一共有8层：5个卷积层和3个全连接层。并且，作者发现删除任何一个卷积层都会导致模型性能的下降。

模型一共训练了5~6天，使用了两块GTX 580 3GB的GPU。并且作者认为如果有更快的GPU和更大的数据集，该模型的结果还会进一步提升。

# 2.数据集（The Dataset）

使用的数据集为ImageNet。ImageNet数据集包含大约1500万张图片，图片中的目标被标记为约22000种不同的类别。

从2010年开始，每年都会举办ILSVRC（ImageNet Large-Scale Visual Recognition Challenge）比赛。ILSVRC通常会从ImageNet中选出一个子集作为比赛的数据集。ILSVRC所用的数据集通常包含1000个类别，每个类别大约1000张图片。总的来说，ILSVRC所用的数据集大约会包含120万张图片用于训练，5万张图片用于验证，15万张图片用于测试。除了ILSVRC-2010，其余年份的ILSVRC均不提供测试集的标签。ILSVRC通常统计两种错误率：Top-1错误率和Top-5错误率。Top-1错误率：针对一个样本，如果模型预测概率最大的结果为正确结果，则该样本被统计为预测正确。Top-5错误率：针对一个样本，如果模型预测概率排名前5的结果中包含正确结果，则该样本即被统计为预测正确。

ImageNet包含不同分辨率的图片，但是AlexNet要求输入的维度是固定的。因此，统一将图片下采样为$256 \times 256$的分辨率。采用的下采样方式：先将图片rescale，使其短边为256（长边采用同样的比例缩放），在此基础上，在图片的中心位置截取$256\times 256$大小的图片作为最终的输入。除此之外，还对网络输入（即$256\times 256$的RGB图片）进行了[归一化](http://shichaoxin.com/2020/02/03/深度学习基础-第十二课-归一化输入/)处理。其他的便没有什么预处理步骤了。

# 3.网络结构（The Architecture）

AlexNet的结构如下图所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/AlexNet/1.png)

以下是AlexNet相较于之前的神经网络架构，比较创新和重要的特点。

## 3.1.ReLU Nonlinearity

在AlexNet之前的神经网络架构通常使用sigmoid或者tanh作为激活函数。在AlexNet中，提出使用ReLU函数作为激活函数，模型可以更快的收敛：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/AlexNet/2.png)

上图中，实线为使用ReLU函数作为激活函数；虚线为使用tanh函数作为激活函数。前者比后者快了6倍左右。

## 3.2.Training on Multiple GPUs

因为训练数据过于庞大（120万训练数据），因此使用两块GTX 580 GPU并行运算。

## 3.3.Local Response Normalization

虽然ReLU函数并不要求对输入进行归一化处理，但是AlexNet的作者发现在某些特定层使用局部响应归一化（Local Response Normalization，LRN）可以提升模型性能（top-1错误率下降1.4%，top-5错误率下降1.2%）。LRN的公式及示意图见下：

$$b^i_{x,y}=\frac{a^i_{x,y}}{( k+ \alpha \sum _{j=\max (0,i-n/2)}^{\min (N-1,i+n/2)} (a^i_{x,y})^2 )^{\beta}}$$

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/AlexNet/3.png)

$a$为经过激活函数得到的值，$i$表示第$i$个通道，$(x,y)$为坐标位置。$b$为$a$经过LRN后得到的值。$N$为通道的总数目。

$k,\alpha,\beta,n$均为超参数。可以通过$n$来决定具体需要累加多少个在i附近的通道的值用于LRN。

AlexNet作者所用的超参数的值为$k=2,n=5,\alpha=10^{-4},\beta=0.75$。

此外，作者还将LRN用于CIFAR-10数据集进行测试：一个四层的CNN在使用了LRN后，错误率从13%降为11%。

## 3.4.Overlapping Pooling

我们现在通常用的pooling都是没有重叠的，例如核大小为$2\times 2$，步长为2。但是AlexNet使用的pooling是有重叠的，核大小为$3\times 3$，步长为2。Overlapping Pooling使得其top-1错误率下降0.4%，top-5错误率下降0.3%。

## 3.5.Overall Architecture

|层数|输入大小|kernel大小|kernel数量|stride|padding|输出大小|备注|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|CONV1|$224\times 224\times 3$|$11\times 11\times 3$|96|4|2|$55\times 55\times 96$|备注1|
|POOL1|$55\times 55\times 96$|$3\times 3$|1|2|0|$27\times 27\times 96$|备注2|
|CONV2|$27\times 27\times 96$|$5\times 5\times 96$|256|1|2|$27\times 27\times 256$|备注3|
|POOL2|$27\times 27\times 256$|$3\times 3$|1|2|0|$13\times 13\times 256$|$\times$|
|CONV3|$13\times 13\times 256$|$3\times 3\times 256$|384|1|1|$13\times 13\times 384$|$\times$|
|CONV4|$13\times 13\times 384$|$3\times 3\times 384$|384|1|1|$13\times 13\times 384$|$\times$|
|CONV5|$13\times 13\times 384$|$3\times 3\times 384$|256|1|1|$13\times 13\times 256$|$\times$|
|POOL5|$13\times 13\times 256$|$3\times 3$|1|2|0|$6\times 6 \times 256$|$\times$|
|FC6|9216|$\times$|$\times$|$\times$|$\times$|4096|备注4|
|FC7|4096|$\times$|$\times$|$\times$|$\times$|4096||
|FC8|4096|$\times$|$\times$|$\times$|$\times$|1000|备注5|

👉备注1：也有好多人将输入修正为$227\times 227\times 3$，然后不做padding。padding列的值指的是填充的圈数。

👉备注2：pooling的方式为max pooling+overlapping pooling。表中剩余的pooling方式与之相同，不再特别说明。

👉备注3：模型是在两个GPU上同时训练的。因此对于一块GPU来说，其输入大小其实是$27\times 27\times 48$，所用的kernel大小是$5\times 5\times 48$，kernel数量为128，stride=1，padding=2，得到的输出大小为$27\times 27\times 128$。表中直接标注的就是两块GPU合起来的结果维度。剩余的层也是这种情况，不再一一说明。

👉备注4：$6\times 6\times 256=9216$。

👉备注5：FC8即为输出层。因为共有1000个类别，所以输出层使用一个1000维度的softmax函数。

# 4.降低过拟合（Reducing Overfitting）

AlexNet共有六千万左右的参数，但是ILSVRC提供的数据集并不能保证训练这么多的参数而不出现过拟合。因此，AlexNet采用了两种减少过拟合的方法。

## 4.1.数据增强（Data Augmentation）

数据增强是指人为的创造一些数据以扩充原有数据集，是一种简单且常用的降低过拟合的方法。AlexNet采用了两种数据增强的方式：

👉图像平移和水平翻转(translations and horizontal reflections)。在$256\times 256$的图像中随机选取$224\times 224$的图像块(patches)及其水平翻转图像，这一操作使得训练集被扩大了2048倍（$2\times (256-224)^2=2048$）。这也是AlexNet输入维度是$224\times 224$的原因。在预测阶段，对于测试图像也选取5个$224\times 224$的图像块（4个角块和1个中心块）及其水平翻转图像，因此一幅测试图像会被扩展为10幅，求这10幅图像预测结果（即softmax层的输出）的均值作为该测试图像的最终预测结果。

👉改变训练集中RGB图像的像素值。首先对整个训练集图像的RGB三个通道的像素值做[PCA](http://shichaoxin.com/2020/09/21/数学基础-第十六课-主成分分析/)，即PCA算法的输入维度为$X_{3\times m}$。对于每个图像中每一点的像素值$I_{xy}=[I^R_{xy}, I^G_{xy}, I^B_{xy}]^T$，我们加上以下值（下式算出来的值维度为$3\times 1$，因为特征向量$p_i$的维度为$3\times 1$。这样刚好可以和像素值的维度$3\times 1$进行对应元素的加法运算）：

$$[\mathbf p_1,\mathbf p_2,\mathbf p_3][\alpha_1 \lambda_1,\alpha_2 \lambda_2,\alpha_3 \lambda_3]^T$$

其中，$p_i$和$\lambda _i$分别是$3\times 3$的RGB协方差矩阵的第$i$个特征向量和第$i$个特征值。$\alpha _i$是一个服从均值为0，标准差为0.1的高斯分布的随机值。每幅图像只随机选择一次$\alpha _i$（如果这幅图象被再次使用，则会重新随机选择一次$\alpha _i$）。这个操作使得AlexNet模型降低了对图像亮度和颜色的敏感度，将top-1错误率降低了1%以上。

## 4.2.Dropout

结合许多不同模型的预测结果是提高模型准确率的一种非常成功的方法，但是对于训练耗时太长的模型来说，这样做的成本太高了。因此，AlexNet提出了一种全新的方法：[dropout](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#5dropout正则化)。它会以50%的概率将隐藏层的神经元输出置为0。以这种方法被置0的神经元不参与网络的前馈和反向传播。因此，每次给网络提供了输入后，神经网络都会采用一个不同的结构，但是这些结构都共享权重。AlexNet在FC6、FC7两个层中使用了dropout。

# 5.训练细节（Details of learning）

AlexNet使用[mini-batch梯度下降法](http://shichaoxin.com/2020/02/20/深度学习基础-第十五课-mini-batch梯度下降法/)+[Momentum梯度下降法](http://shichaoxin.com/2020/03/05/深度学习基础-第十七课-Momentum梯度下降法/)。其中，batch-size=128，momentum系数为0.9，[权重衰减(weight decay)](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#314l1正则化和l2正则化的区别)为0.0005。AlexNet作者发现这种较小的权重衰减对于模型的训练很重要。换句话说，权重衰减在这里不仅仅是一个正则化方法，它还减少了模型的训练误差。权重更新的方法见下：

$$v_{i=1} := 0.9 \cdot v_i -0.0005 \cdot \epsilon \cdot w_i - \epsilon \cdot \langle \frac{\partial L}{\partial w} \mid _{w_i} \rangle _{D_i}$$

$$w_{i+1}:=w_i + v_{i+1}$$

其中，$i$表示当前的迭代次数，$v$表示momentum，$\epsilon$表示学习率，$\langle \frac{\partial L}{\partial w} \mid _{w_i} \rangle _{D_i}$是第$i$批次的目标函数关于$w$的导数（$w_i$的偏导数）$D_i$的平均值。

AlexNet使用标准差为0.01，均值为0的高斯分布来初始化各层的权重。使用常数1初始化网络的第二、第四和第五个卷积层以及所有全连接层的偏置项。使用常数0初始化剩余层的偏置项。

AlexNet对所有层使用一样的学习率，但是在训练过程中会对学习率进行手动调整。学习率初始值设置为0.01。学习率的调整方式：以当前的学习率进行训练，当验证集上的错误率停止降低时，将学习率除以10。

AlexNet作者在训练时，一共进行了3次学习率的调整，训练集使用了120万张图像，迭代了90次左右，在两块NVIDIA GTX 580 3GB的GPU上训练了5~6天的时间。

# 6.结果（Results）

论文本节内容主要是在展示AlexNet的预测成果以及其取得的成绩，因此不再详述。

ILSVRC2010比赛冠军方法是Sparse coding，AlexNet与其比较：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/AlexNet/4.png)

ILSVRC-2012，AlexNet参加比赛，获得冠军，远超第二名SIFT+FVs：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/AlexNet/5.png)

# 7.讨论（Discussion）

AlexNet的成功说明了深度对于神经网络的重要性。

# 8.原文链接

👽[ImageNet Classification with Deep Convolutional Neural Networks](https://github.com/x-jeff/AI_Papers/blob/master/ImageNet%20Classification%20with%20Deep%20Convolutional%20Neural%20Networks.pdf)

# 9.参考资料

1. [Alexnet中的LRN](https://www.pianshen.com/article/71331232633/)
2. [AlexNet论文解读]( https://zhuanlan.zhihu.com/p/157643267)