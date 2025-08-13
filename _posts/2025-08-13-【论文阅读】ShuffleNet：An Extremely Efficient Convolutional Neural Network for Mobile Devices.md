---
layout:     post
title:      【论文阅读】ShuffleNet：An Extremely Efficient Convolutional Neural Network for Mobile Devices
subtitle:   ShuffleNet，channel shuffle
date:       2025-08-13
author:     x-jeff
header-img: blogimg/20220130.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

提出了极具计算效率的CNN架构：ShuffleNet，专为计算能力非常有限的移动设备设计。该新架构采用了两种新操作：pointwise group convolution和channel shuffle，以在保证准确率的同时大幅降低计算成本。

# 2.Related Work

不再详述。

# 3.Approach

## 3.1.Channel Shuffle for Group Convolutions

CNN通常由重复的block组成，这些block具有相同的结构。其中，一些网络，比如Xception和[ResNeXt](https://shichaoxin.com/2023/12/11/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Aggregated-Residual-Transformations-for-Deep-Neural-Networks/)，在block中引入了高效的[深度分离卷积](https://shichaoxin.com/2024/12/25/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-MobileNets-Efficient-Convolutional-Neural-Networks-for-Mobile-Vision-Applications/#31depthwise-separable-convolution)或分组卷积（group convolutions），以在性能和计算成本之间取得出色的平衡。然而，我们注意到，这两种设计并未充分考虑$1 \times 1$卷积（亦称pointwise convolutions），而$1\times 1$卷积需要相当高的计算复杂度。例如，在[ResNeXt](https://shichaoxin.com/2023/12/11/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Aggregated-Residual-Transformations-for-Deep-Neural-Networks/)中，只有$3\times 3$卷积层使用了分组卷积。因此，在[ResNeXt](https://shichaoxin.com/2023/12/11/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Aggregated-Residual-Transformations-for-Deep-Neural-Networks/)的每个残差单元中，pointwise convolutions占据了93.4%的乘加运算量（cardinality=32）。

>这里用一张图简单解释下分组卷积：
>
>![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/1.png)

为了解决这个问题，一个直接的方法就是在$1\times 1$卷积层上也使用通道稀疏连接，比如分组卷积。分组卷积可以显著降低计算成本，然而，如果多个分组卷积堆叠在一起，就会出现一个副作用：某个通道的输出仅来源于输入通道中的一小部分，如Fig1(a)所示，很明显，某个分组的输出只与该分组内的输入有关。这一特性阻碍了通道分组之间的信息流动。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/2.png)

Fig1展示的是两个分组卷积堆叠在一起的情况，GConv表示Group Conv。

因此，我们提出了channel shuffle，如Fig1(b)所示。其高效实现可用下图来表示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/3.png)

channel shuffle是可微分的，这意味着它可以嵌入到网络结构中进行端到端的训练。

## 3.2.ShuffleNet Unit

利用channel shuffle，我们提出了一个专为小型网络设计的ShuffleNet unit。如Fig2所示，Fig2(a)就是一个常规的bottleneck unit，DWConv表示[depthwise convolution](https://shichaoxin.com/2024/02/25/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-RTMDet-An-Empirical-Study-of-Designing-Real-Time-Object-Detectors/#32model-architecture)。在Fig2(b)中，我们在第一个$1\times 1$分组卷积之后接了个channel shuffle，第二个$1\times 1$分组卷积是为了调整通道数以匹配shortcut path，在这里，为了简化，我们在第二个$1\times 1$分组卷积之后并没有接channel shuffle。注意，在Fig2(b)和Fig2(c)中，DWConv之后，只有[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)，没有[ReLU](https://shichaoxin.com/2019/12/11/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E8%AF%BE-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/#22relu%E5%87%BD%E6%95%B0)。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/4.png)

由于引入了channel shuffle，ShuffleNet unit的计算非常高效。和[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)、[ResNeXt](https://shichaoxin.com/2023/12/11/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Aggregated-Residual-Transformations-for-Deep-Neural-Networks/)相比，一样的设置下，ShuffleNet的计算复杂度更低。

## 3.3.Network Architecture

ShuffleNet的整体框架见下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/5.png)

我们使用缩放因子$s$来控制通道数量。将表1中的网络记为"ShuffleNet $1\times$"，使用"ShuffleNet $s\times$"表示将通道数量缩放$s$倍，对应的计算成本会是"ShuffleNet $1\times$"的大约$s^2$倍。

# 4.Experiments

我们在ImageNet 2012分类数据集上进行了评估。训练设置和超参基本和[ResNeXt](https://shichaoxin.com/2023/12/11/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Aggregated-Residual-Transformations-for-Deep-Neural-Networks/)保持一致，只有两处不同：1）将weight decay设置为$4e-5$，使用线性学习率衰减策略（从0.5降到0）；2）未使用过于激进的数据增强策略。用了4块GPU，batch size=1024，一共训练了$3\times 10^5$次迭代，大约用时1到2天。至于benchmark，我们在ImageNet验证集上比较了single crop top-1性能，即在$256\times$大小的输入图像的中心裁剪一个$224\times 224$大小的crop，以此来评估分类准确率。

## 4.1.Ablation Study

### 4.1.1.Pointwise Group Convolutions

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/6.png)

### 4.1.2.Channel Shuffle vs. No Shuffle

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/7.png)

## 4.2.Comparison with Other Structure Units

为了和其他结构单元公平比较，我们依照表1的网络结构，将Stage 2-4的ShuffleNet units替换为其他结构单元，并通过调整通道数量使计算成本基本保持不变。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/8.png)

## 4.3.Comparison with MobileNets and Other Frameworks

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/9.png)

>SE见文献：J. Hu, L. Shen, and G. Sun. Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 2017.。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/10.png)

## 4.4.Generalization Ability

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/11.png)

## 4.5.Actual Speedup Evaluation

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ShuffleNet/12.png)

# 5.原文链接

👽[ShuffleNet：An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://github.com/x-jeff/AI_Papers/blob/master/2025/ShuffleNet%EF%BC%9AAn%20Extremely%20Efficient%20Convolutional%20Neural%20Network%20for%20Mobile%20Devices.pdf)