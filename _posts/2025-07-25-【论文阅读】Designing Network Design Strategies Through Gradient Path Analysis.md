---
layout:     post
title:      【论文阅读】Designing Network Design Strategies Through Gradient Path Analysis
subtitle:   ELAN
date:       2025-07-25
author:     x-jeff
header-img: blogimg/20210414.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

通过Fig1中的分析，我们发现通过调整训练目标与损失层的配置，可以控制每一层（无论浅层还是深层）学习到的特征类型。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/1.png)

Fig1主要是想说明在深度神经网络中，无论是浅层还是深层，它们都有提取低级特征或高级特征的能力。

我们的主要观点是目标函数能够引导神经网络学习信息。我们知道目标函数是通过梯度的反向传播来更新每层的权重。因此，我们可以根据梯度反向传播的路径来设计网络结构，一共分为三个不同层级的设计策略：

1. **Layer-level design：**
    * 参见[PRN](https://shichaoxin.com/2025/04/24/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Enriching-Variety-of-Layer-wise-Learning-Information-by-Gradient-Combination/)。
2. **Stage-level design：**
    * 参见[CSPNet](https://shichaoxin.com/2023/12/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-CSPNET-A-NEW-BACKBONE-THAT-CAN-ENHANCE-LEARNING-CAPABILITY-OF-CNN/)。
3. **Network-level design：**
    * 详见第2.4部分提出的**ELAN（Efficient Layer Aggregation Network）**。

# 2.Methodology

## 2.1.Network Design Strategies

如Fig2所示，在本文中，我们将网络设计策略分为两类：

1. **数据路径设计策略：**
    * 数据路径设计策略主要关注于特征提取、特征选择以及特征融合操作的设计，以提取具有特定属性的特征。这些特征可以帮助后续网络层进一步利用这些信息，获取更优的特性用于更高级的分析。
    * 优点：1）能够提取具有特定物理意义的特征；2）可以针对不同输入，利用参数化模型自动选择合适的运算单元；3）所学特征可被直接复用。
    * 缺点：1）在训练过程中，可能会出现无法预测的性能退化，此时需要设计更复杂的网络架构来解决；2）多种专门设计的计算单元容易导致性能优化困难。
2. **梯度路径设计策略：**
    * 梯度路径设计策略的目的是分析梯度的来源和构成方式，以及它们如何被驱动参数所更新。基于上述分析结果，可以据此设计网络结构。该设计理念希望实现更高的参数利用率，从而达到更优的学习效果。
    * 优点：1）能够高效利用网络参数；2）具备稳定的模型学习能力；3）推理速度快。
    * 缺点：当梯度更新路径不再是网络的简单反向前馈路径时，编程的复杂性将大大增加。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/2.png)

## 2.2.Partial Residual Networks

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/3.png)

详见[PRN](https://shichaoxin.com/2025/04/24/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Enriching-Variety-of-Layer-wise-Learning-Information-by-Gradient-Combination/)。

## 2.3.Cross Stage Partial Networks

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/4.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/5.png)

详见[CSPNet](https://shichaoxin.com/2023/12/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-CSPNET-A-NEW-BACKBONE-THAT-CAN-ENHANCE-LEARNING-CAPABILITY-OF-CNN/)。

## 2.4.Efficient Layer Aggregation Networks

设计ELAN的主要目的是为了解决在模型扩展（model scaling）过程中，收敛性会逐渐恶化的问题。

当我们进行模型扩展时，会出现一种现象：当网络深度达到某一临界值后，如果我们继续在计算模块中堆叠结构，准确率的提升将变得越来越小，甚至没有提升。更糟糕的是，当网络达到某个关键深度时，其收敛性能开始恶化，导致整体准确率甚至低于浅层网络。其中一个典型的例子是[scaled-YOLOv4](https://shichaoxin.com/2025/07/15/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Scaled-YOLOv4-Scaling-Cross-Stage-Partial-Network/)，我们可以看到其[P7模型](https://shichaoxin.com/2025/07/15/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Scaled-YOLOv4-Scaling-Cross-Stage-Partial-Network/)虽然使用了大量参数和计算操作，但准确率的提升却非常有限。而这种现象也普遍出现在许多流行的网络中。例如，[ResNet-152](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)的计算复杂度大约是[ResNet-50](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)的三倍，但在ImageNet上的准确率提升却不到1%。而当[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)堆叠到200层时，其准确率甚至比[ResNet-152](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)还差。同样地，当[VoVNet](https://shichaoxin.com/2025/04/14/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-An-Energy-and-GPU-Computation-Efficient-Backbone-Network-for-Real-Time-Object-Detection/)堆叠到99层时，其准确率远低于[VoVNet-39](https://shichaoxin.com/2025/04/14/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-An-Energy-and-GPU-Computation-Efficient-Backbone-Network-for-Real-Time-Object-Detection/)。从梯度路径设计策略的角度出发，我们推测[VoVNet](https://shichaoxin.com/2025/04/14/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-An-Energy-and-GPU-Computation-Efficient-Backbone-Network-for-Real-Time-Object-Detection/)的准确率下降比[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)更快的原因在于：[VoVNet](https://shichaoxin.com/2025/04/14/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-An-Energy-and-GPU-Computation-Efficient-Backbone-Network-for-Real-Time-Object-Detection/)的堆叠是基于[OSA模块](https://shichaoxin.com/2025/04/14/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-An-Energy-and-GPU-Computation-Efficient-Backbone-Network-for-Real-Time-Object-Detection/)的。我们知道，每个[OSA模块](https://shichaoxin.com/2025/04/14/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-An-Energy-and-GPU-Computation-Efficient-Backbone-Network-for-Real-Time-Object-Detection/)都包含一个transition layer，所以每当我们堆叠一个[OSA模块](https://shichaoxin.com/2025/04/14/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-An-Energy-and-GPU-Computation-Efficient-Backbone-Network-for-Real-Time-Object-Detection/)时，整个网络中每一层的最短梯度路径都会增加1。而对于[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)，其堆叠结构是残差模块，残差模块的堆叠只会增加最长梯度路径，不会增加最短梯度路径。为了验证模型扩展带来的这些可能问题，我们基于[YOLOR-CSP](https://shichaoxin.com/2025/07/18/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-You-Only-Learn-One-Representation-Unified-Network-for-Multiple-Tasks/)进行了实验。从实验结果来看，当堆叠层数达到80层以上时，CSP fusion first的准确率开始超过普通的[CSPNet](https://shichaoxin.com/2023/12/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-CSPNET-A-NEW-BACKBONE-THAT-CAN-ENHANCE-LEARNING-CAPABILITY-OF-CNN/)。此时，每个stage的计算模块的最短梯度路径减少了1。随着网络进一步加深，CSP fusion last将获得最高的准确率，但此时整个网络所有层的最短梯度路径减少了1。这些实验结果验证了我们之前的假设。在这些实验的支持下，我们设计了ELAN，如Fig6所示。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/6.png)

下面是一个更加详细的ELAN结构图：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/18.png)

ELAN block的输入是$X_{input}$，然后分为两个分支：short branch和main branch。

对于short branch，其充当着cross-stage connection的角色，通过使用一个$1\times 1$卷积来降低通道数量，但并不改变feature map的大小：

$$X_{short} = f_{CBS(ks=1\times 1)}(X_{input})$$

$f_{CBS}$表示非线性转换函数，结合了卷积层、[SiLU](https://shichaoxin.com/2022/04/09/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-GAUSSIAN-ERROR-LINEAR-UNITS-(GELUS)/)层和[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)层。

对于main branch，先是一个$1\times 1$卷积，然后级联$k$个卷积模块，每个卷积模块包含$m$个卷积层：

$$\begin{align*} X_{main} &= f_{CBS(ks=1\times 1)}(X_{input}) \\ X_k &= F_{ConvModule}(X_{k-1}) \\&= f^1_{CBS(ks=3\times 3)}(...(f^m_{CBS(ks=3\times 3)}(X_{k-1}))) \end{align*}$$

最终block的输出可表示为：

$$X_{out}= [X_{short};X_{main};X_I;...;X_k]$$

# 3.Analysis

本部分，我们将基于经典网络架构分析所提出的梯度路径设计策略。

## 3.1.Analysis of gradient combination

研究人员通常使用最短梯度路径（是指从损失函数传回梯度到某一层的最短路径长度）和集成特征数量（表示在每层中能整合来自多少先前层的特征）来衡量网络架构的学习效率与能力。但是从表1来看，这些指标与准确率和参数使用量之间并没有完全的关联。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/7.png)

我们发现用于更新不同层权重的梯度组合（gradient combinations）与网络的学习能力密切相关。梯度组合分为两部分：梯度时间戳（Gradient Timestamp）和梯度来源（Gradient Source）。

这部分的内容可以参阅：[Combination of Gradients](https://shichaoxin.com/2025/04/24/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Enriching-Variety-of-Layer-wise-Learning-Information-by-Gradient-Combination/#3combination-of-gradients)。从内容到配图，几乎是一样的，Fig7对应[这里的Fig5](https://shichaoxin.com/2025/04/24/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Enriching-Variety-of-Layer-wise-Learning-Information-by-Gradient-Combination/#31timestamp)，Fig8对应[这里的Fig6](https://shichaoxin.com/2025/04/24/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Enriching-Variety-of-Layer-wise-Learning-Information-by-Gradient-Combination/#32source)。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/8.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/9.png)

## 3.2.Analysis of cross stage partial strategy

[CSPNet](https://shichaoxin.com/2023/12/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-CSPNET-A-NEW-BACKBONE-THAT-CAN-ENHANCE-LEARNING-CAPABILITY-OF-CNN/)成功的将梯度组合的概念和硬件资源利用效率相结合，从而使所设计的网络结构同时提升了学习能力和推理速度。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/10.png)

## 3.3.Analysis of length of gradient path

[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)中的残差块除了通过计算块（computational block）传播梯度外，还有一部分梯度会通过恒等连接（identity connection）传播。因此，每个残差块中同时存在两条梯度路径。在这里，我们分别对计算块和恒等连接施加梯度停止（stop gradient）操作，如Fig9所示。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/11.png)

实验结果见表3：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/12.png)

实验结果表明，在[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)中缩短梯度路径确实是提升超深网络收敛性的关键因素。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/13.png)

# 4.Experiments

## 4.1.Experimental setup

不再详述。

## 4.2.Layer-level gradient path design strategies

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/14.png)

## 4.3.Stage-level gradient path design strategies

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/15.png)

## 4.4.Network-level gradient path design strategies

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/16.png)

## 4.5.Comparison

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/ELAN/17.png)

# 5.Conclusions

不再赘述。

# 6.原文链接

👽[Designing Network Design Strategies Through Gradient Path Analysis](https://github.com/x-jeff/AI_Papers/blob/master/2025/Designing%20Network%20Design%20Strategies%20Through%20Gradient%20Path%20Analysis.pdf)

# 7.参考资料

1. [Detection of Military Targets on Ground and Sea by UAVs with Low-Altitude Oblique Perspective](https://www.researchgate.net/publication/379681440_Detection_of_Military_Targets_on_Ground_and_Sea_by_UAVs_with_Low-Altitude_Oblique_Perspective)