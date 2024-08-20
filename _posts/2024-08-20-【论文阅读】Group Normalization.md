---
layout:     post
title:      【论文阅读】Group Normalization
subtitle:   Batch Norm，Layer Norm，Instance Norm，Group Norm
date:       2024-08-20
author:     x-jeff
header-img: blogimg/20210224.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)已经被确认是深度学习中非常有效的一个组成部分。[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)通过计算batch内的均值和方差来归一化特征。

但是，[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)需要较大的batch size才能良好的工作，如果减少[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的batch size会大大增加模型误差（如Fig1所示）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/1.png)

>在本文中，batch size指的是每个worker（即GPU）处理的样本数量。[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的计算是基于每个worker的，并不会跨worker计算，很多库都是按照这种标准实现的。

本文提出GN（Group Normalization）作为[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的一种简单替代方案。我们注意到，许多经典特征，如[SIFT](https://shichaoxin.com/2022/12/29/OpenCV%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%89%E5%8D%81%E5%85%AD%E8%AF%BE-SIFT%E7%89%B9%E5%BE%81%E6%A3%80%E6%B5%8B/)和[HOG](https://shichaoxin.com/2023/09/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Histograms-of-Oriented-Gradients-for-Human-Detection/)，都是group-wise的特征，进行group-wise的归一化。类似的，GN将通道划分为组，并在组内进行特征归一化，如Fig2所示。GN的计算与batch size无关。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/2.png)

Fig2展示了4种不同的特征归一化方法，每个子图的三个轴分别代表batch size（$N$）、通道数（$C$）和feature map大小（$H \times W$）。蓝色区域内的子块在归一化时使用相同的均值和方差，其中均值和方差的计算基于整个蓝色区域内的所有子块。下面通过几个更直观的图来进一步说明下这4种特征归一化方法，其中颜色相同的部分表示使用相同的均值和方差进行归一化：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/3.png)

如Fig1所示，不同batch size下，GN的表现非常稳定。

如Fig2所示，[LN（Layer Normalization）](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)和IN（Instance Normalization）也都和batch size无关。[LN](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)和IN对训练序列模型（[RNN](https://shichaoxin.com/2020/11/22/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E5%9B%9B%E5%8D%81%E8%AF%BE-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/)/[LSTM](https://shichaoxin.com/2020/12/09/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E5%9B%9B%E5%8D%81%E4%BA%8C%E8%AF%BE-GRU%E5%92%8CLSTM/#3lstm)）和生成式模型（[GANs](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/)）非常有效。但是根据我们的实验，在CV任务中，[LN](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)和IN的效果不如GN。反过来，GN也可以代替[LN](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)和IN用于序列或生成式模型。

# 2.Related Work

不再赘述。

# 3.Group Normalization

## 3.1.Formulation

一系列的特征归一化方法，包括[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)、[LN](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)、IN、GN，都会执行如下计算：

$$\hat{x}_i = \frac{1}{\sigma_i}(x_i - \mu_i) \tag{1}$$

其中，$x$是计算得到的特征，$i$是索引。对于2D图像，$i$是一个4维向量，即$i=(i_N,i_C,i_H,i_W)$，其是$(N,C,H,W)$的索引，其中，$N$是batch，$C$是通道，$H,W$是feature map的高和宽。

式(1)中的$\mu$和$\sigma$分别是均值和标准差，计算公式为：

$$\mu_i = \frac{1}{m} \sum_{k \in S_i} x_k, \  \sigma_i = \sqrt{\frac{1}{m}\sum_{k \in S_i}(x_k - \mu_i)^2+\epsilon} \tag{2}$$

其中，$\epsilon$是一个小的常数。$S_i$是用于计算均值和标准差的像素点集合，$m$是这个集合中像素点的数目。如Fig2所示，多数特征归一化的差异就在于$S_i$的选取。

在[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)中，$S_i$的定义为：

$$S_i = \{k \mid k_C = i_C \} \tag{3}$$

上式表示位于同一通道的所有像素点会被一起归一化，即对于每个通道，沿着$(N,H,W)$轴计算$\mu$和$\sigma$。在[LN](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)中，$S_i$定义为：

$$S_i = \{k \mid k_N = i_N \} \tag{4}$$

即对于每个样本，沿着$(C,H,W)$轴计算$\mu$和$\sigma$。在IN中，$S_i$的定义为：

$$S_i = \{ k \mid k_N = i_N, k_C = i_C \} \tag{5}$$

即对于每个样本的每个通道，沿着$(H,W)$轴计算$\mu$和$\sigma$。

此外，对于[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)、[LN](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)和IN，针对每个通道都会学习一个线性变换，以补偿可能丧失的表征能力：

$$y_i = \gamma \hat{x}_i + \beta \tag{6}$$

其中，$\gamma$和$\beta$是可训练的scale和shift。

GN的$S_i$定义为：

$$S_i = \{ k \mid k_N=i_N, \lfloor \frac{k_C}{C/G} \rfloor = \lfloor \frac{i_C}{C/G} \rfloor \} \tag{7}$$

其中，$G$是组数，是一个预先定义好的超参数（默认$G=32$）。$C/G$是每组的通道数量。如Fig2最右所示，有$G=2$，每个组包含3个通道。

给定式(7)中的$S_i$，可以通过式(1)、式(2)和式(6)来定义GN层。具体来说，同组内的像素点使用相同的$\mu$和$\sigma$进行归一化。针对每个通道，GN也学习$\gamma$和$\beta$。

如果$G=1$，则GN就等同于[LN](https://shichaoxin.com/2022/03/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Layer-Normalization/)。如果$G=C$，则GN就等同于IN。

## 3.2.Implementation

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/4.png)

# 4.Experiments

## 4.1.Image Classification in ImageNet

我们在ImageNet分类数据集（1000个类别）上进行了实验。在~1.28M张图像上进行了训练，在50,000张验证图像上进行了评估，模型使用[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)。

👉**Implementation details.**

训练所有的模型都是用了8块GPU，在每块GPU内，[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)计算均值和方差。使用论文“K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.”中的方法去初始化所有模型的所有卷积层。我们用1来初始化所有的$\gamma$参数，除了每个残差块的最后一个归一化层，我们用0初始化$\gamma$。对于包括$\gamma,\beta$在内的所有权重层，都有weight decay=0.0001。所有的模型都训练了100个epoch，分别在第30、60、90个epoch时，学习率除以10。训练阶段使用了data augmentation。在验证集上，我们使用中心裁剪的$224 \times 224$大小的图像来评估top-1分类误差。为了减少随机变化，我们报告了最后5个epoch的中值错误率。其他细节见“S. Gross and M. Wilber. Training and investigating Residual Nets. https://github.com/facebook/fb.resnet.torch, 2016.”。

baseline是[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)+[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)。所有模型的超参数都一样。

👉**Comparison of feature normalization methods.**

首先实验常规的batch size=32张图像（每个GPU）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/5.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/6.png)

👉**Small batch sizes.**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/7.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/8.png)

👉**Comparison with Batch Renorm (BR).**

>BR：S. Ioffe. Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. In NIPS, 2017.。

BR引入了两个额外的参数（$r$和$d$）来约束[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)估计的均值和方差。它们的值受到$r_{max}$和$d_{max}$的控制。对于[ResNet-50](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)+BR，我们设$r_{max}=1.5,d_{max}=0.5$。当batch size=4时，BR的错误率为26.3%，优于[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的27.3%，但是差于GN的24.2%。

👉**Group division.**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/9.png)

👉**Deeper models.**

也在[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)上测试了[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)和GN。当batch size=32时，[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的验证错误率为22.0%，GN是22.4%。当batch size=2时，[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的错误率是31.9%，而GN是23.0%。

👉**Results and analysis of VGG models.**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/10.png)

## 4.2.Object Detection and Segmentation in COCO

接下来我们评估在目标检测和分割上的fine-tune模型。对于这种CV任务，输入图像的分辨率通常都很高，所以batch size一般都比较小：1 image/GPU或2 images/GPU。作为结果，[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)变成了一个线性层：$y=\frac{\gamma}{\sigma}(x-\mu)+\beta$，其中$\mu,\sigma$是预训练模型事先计算好的，它们不再更新。我们将其记为BN\*，表示在fine-tune阶段其实并没有执行归一化。我们也尝试了在fine-tune阶段正常更新参数$\mu$和$\sigma$，但效果很差，在batch size=2的情况下，AP降低了6个点，所以我们放弃了这种方案。

使用[Mask R-CNN](https://shichaoxin.com/2023/12/25/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Mask-R-CNN/)作为baseline。在ImageNet上进行了预训练，在fine-tune时，将BN\*替换为GN。在fine-tune阶段，对于参数$\gamma$和$\beta$，设weight decay=0。在fine-tune时，设batch size为1 image/GPU，使用8块GPU。

在COCO train2017上fine-tune，在COCO val2017上评估。

👉**Results of C4 backbone.**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/11.png)

👉**Results of FPN backbone.**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/12.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/13.png)

👉**Training Mask R-CNN from scratch.**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/14.png)

## 4.3.Video Classification in Kinetics

在Kinetics数据集上评估了视频分类。许多视频分类模型将特征扩展到3D时空维度。这需要大量内存，对batch size和模型设计施加了限制。

我们使用I3D（Inflated 3D）卷积网络。使用ResNet-50 I3D作为baseline。在ImageNet上进行预训练。对于[BN](https://shichaoxin.com/2021/11/02/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)和GN，我们都将归一化从$(H,W)$扩展到了$(T,H,W)$，其中$T$是时间轴。我们在有着400个类别的Kinetics训练集上进行训练，在验证集上进行评估。我们报告了top-1和top-5分类精度，最终结果是10个clips的平均。

我们研究了两个时间长度：32帧和64帧。32帧是从原始视频中每隔两帧采样一帧，64帧是连续采样。64帧消耗的内存是32帧的2倍。对于32帧，batch size=8或4（每个GPU）。对于64帧，batch size=4（每个GPU）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/15.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/GroupNormalization/16.png)

# 5.Discussion and Future Work

不再赘述。

# 6.原文链接

👽[Group Normalization](https://github.com/x-jeff/AI_Papers/blob/master/2024/Group%20Normalization.pdf)

# 7.参考资料

1. [BatchNorm, LayerNorm, InstanceNorm和GroupNorm总结](https://mathpretty.com/11223.html)