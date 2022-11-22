---
layout:     post
title:      【论文阅读】Layer Normalization
subtitle:   Layer Normalization
date:       2022-03-19
author:     x-jeff
header-img: blogimg/20220319.jpg
catalog: true
tags:
    - Natural Language Processing
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Abstract

>本博文只介绍原文的摘要和第3部分，原文链接在本文末尾。

训练SOTA的深度神经网络的计算成本都非常高。一个减少训练时间的方法是normalize神经元的激活值（activities）。比如[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)就显著减少了前馈神经网络的训练时间。但是[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的效果取决于mini-batch size，并且其难以应用在[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)当中。在本文中，我们将[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)转换为Layer Normalization，只利用单个训练样本在一层内的均值和方差。和[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)类似，我们也使用了$\gamma$（本文描述为gain）和$\beta$（本文描述为bias）。和[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)不同的是，Layer Normalization在训练和测试时执行完全相同的计算。并且，Layer Normalization可以直接应用于[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)中的每个时间步（time step）。此外，Layer Normalization对稳定[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)中的hidden state非常有效。Layer Normalization可以显著减少训练时间。

# 2.Layer normalization

我们现在使用Layer Normalization来解决[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)存在的问题。

需要注意的是，上一层输出的变化和下一层输入的变化总是高度相关的，尤其是ReLU函数。这表明我们可以通过固定每一层内总输入的均值和方差来减少“covariate shift”的问题。因此，我们计算同一层内所有隐藏神经元的归一化统计量如下：

$$\mu ^l = \frac{1}{H} \sum^H_{i=1} a_i^l \quad \sigma^l=\sqrt{\frac{1}{H} \sum^H_{i=1} (a_i^l - \mu^l)^2} \tag{3} $$

$l$表示第$l$个隐藏层，$H$为第$l$个隐藏层内隐藏神经元的数量，$a_i^l$为第$l$个隐藏层中第$i$个神经元的输入（进激活函数前）：

$$a_i^l = w_i^{l^\top} h^l \quad h_i^{l+1} = f(a_i^l + b_i^l) \tag{1}$$

其中，$f(\cdot)$为激活函数，$b_i^l$为偏置项。

一层内的所有隐藏神经元都共享一组$\mu$和$\sigma$，但是不同的训练样本有着不一样的归一化统计量（即$\mu$和$\sigma$）。和[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)不同，Layer Normalization不受mini-batch size的约束，且可以在mini-batch size=1的情况下使用。

## 2.1.Layer normalized recurrent neural networks

在NLP中，最近的[Seq2Seq](http://shichaoxin.com/2021/02/23/深度学习基础-第四十六课-Beam-Search/#1seq2seq模型)大多采用[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)来解决序列预测问题。在NLP任务中，对于不同的训练样本，句子的长度不同是很常见的，[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)通过时间步可以很容易的处理这个问题。如果在[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)中使用[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)的话，我们需要为序列中的每个时间步计算并存储单独的统计信息。此时如果一个测试序列比任何一个训练序列都长，就会出现问题。但是Layer Normalization就不会存在这样的问题。并且在Layer Normalization中，所有时间步共用一组gain和bias。

在标准的[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)中，循环层的总输入来自当前输入$\mathbf{x}^t$和上一个hidden state $\mathbf{h}^{t-1}$。总输入$\mathbf{a}^t=W_{hh}\mathbf{h}^{t-1}+W_{xh}\mathbf{x}^t$。Layer Normalization的计算见下：

$$\mathbf{h}^t=f\left[ \frac{\mathbf{g}}{\sigma^t} \odot (\mathbf{a}^t - \mu^t) + \mathbf{b} \right] \quad \mu^t = \frac{1}{H} \sum^H_{i=1} a_i^t \quad \sigma^t = \sqrt{\frac{1}{H} \sum^H_{i=1} (a_i^t - \mu^t)^2} \tag{4}$$

其中，$\odot$为两个向量之间element-wise的乘法。$\mathbf{g},\mathbf{b}$分别代表gain和bias。

Layer Normalization的使用可以在一定程度上解决[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)中的[梯度消失和梯度爆炸](http://shichaoxin.com/2020/02/07/深度学习基础-第十三课-梯度消失和梯度爆炸/)问题。

# 3.原文链接

👽[Layer Normalization](https://github.com/x-jeff/AI_Papers/blob/master/Layer%20Normalization.pdf)