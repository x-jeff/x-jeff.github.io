---
layout:     post
title:      【机器学习基础】第五十四课：[计算学习理论]Rademacher复杂度
subtitle:   Rademacher复杂度
date:       2024-12-15
author:     x-jeff
header-img: blogimg/20211229.jpg
catalog: true
tags:
    - Machine Learning Series
---
>【机器学习基础】系列博客为参考周志华老师的《机器学习》一书，自己所做的读书笔记。  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Rademacher复杂度

基于[VC维](http://shichaoxin.com/2024/10/14/机器学习基础-第五十三课-计算学习理论-VC维/)的泛化误差界是分布无关、数据独立的，也就是说，对任何数据分布都成立。这使得基于[VC维](http://shichaoxin.com/2024/10/14/机器学习基础-第五十三课-计算学习理论-VC维/)的可学习性分析结果具有一定的“普适性”；但从另一方面来说，由于没有考虑数据自身，基于[VC维](http://shichaoxin.com/2024/10/14/机器学习基础-第五十三课-计算学习理论-VC维/)得到的泛化误差界通常比较“松”，对那些与学习问题的典型情况相差甚远的较“坏”分布来说尤其如此。

Rademacher复杂度（Rademacher complexity）是另一种刻画假设空间复杂度的途径，与[VC维](http://shichaoxin.com/2024/10/14/机器学习基础-第五十三课-计算学习理论-VC维/)不同的是，它在一定程度上考虑了数据分布。

给定训练集$D = \\{ (\mathbf{x}_1,y_1),(\mathbf{x}_2,y_2),...,(\mathbf{x}_m,y_m) \\}$，假设$h$的经验误差为：

$$\begin{align} \hat{E}(h) &= \frac{1}{m} \sum_{i=1}^m \mathbb{I}(h(\mathbf{x}_i)\neq y_i) \\&= \frac{1}{m} \sum_{i=1}^m \frac{1-y_i h(\mathbf{x}_i)}{2} \\&= \frac{1}{2} - \frac{1}{2m} \sum_{i=1}^m y_i h(\mathbf{x}_i) \end{align} \tag{1}$$

其中$\frac{1}{m} \sum_{i=1}^m y_i h(\mathbf{x}_i)$体现了预测值$h(\mathbf{x}_i)$与样例真实标记$y_i$之间的一致性，若对于所有$i \in \\{ 1,2,...,m \\}$都有$h(\mathbf{x}_i)=y_i$，则$\frac{1}{m} \sum_{i=1}^m y_i h(\mathbf{x}_i)$取最大值1。也就是说，经验误差最小的假设是：

$$\argmax_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m y_i h(\mathbf{x}_i) \tag{2}$$

然而，现实任务中样例的标记有时会受到噪声影响，即对某些样例$(\mathbf{x}_i,y_i)$，其$y_i$或许已受到随机因素的影响，不再是$\mathbf{x}_i$的真实标记。在此情形下，选择假设空间$\mathcal{H}$中在训练集上表现最好的假设，有时还不如选择$\mathcal{H}$中事先已考虑了随机噪声影响的假设。

考虑随机变量$\sigma_i$，它以0.5的概率取值$-1$，0.5的概率取值+1，称为Rademacher随机变量。基于$\sigma_i$，可将式(2)重写为：

$$\sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i h (\mathbf{x}_i) \tag{3}$$

>$\mathcal{H}$是无限假设空间，有可能取不到最大值，因此使用上确界代替最大值。

考虑$\mathcal{H}$中的所有假设，对式(3)取期望可得：

$$\mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i h (\mathbf{x}_i) \right] \tag{4}$$

其中$\boldsymbol{\sigma} = \\{ \sigma_1,\sigma_2,...,\sigma_m \\}$（个人注解：可多次随机生成）。式(4)的取值范围是$[0,1]$，它体现了假设空间$\mathcal{H}$的表达能力，例如，当$\lvert \mathcal{H} \rvert = 1$时，$\mathcal{H}$中仅有一个假设，这时可计算出式(4)的值为0（个人注解：主要是因为$\boldsymbol{\sigma}$这个随机变量的期望是0）；当$\lvert \mathcal{H} \rvert = 2^m$且$\mathcal{H}$能打散$D$时，对任意$\boldsymbol{\sigma}$总有一个假设使得$h(\mathbf{x}_i) = \sigma_i \  (i=1,2,...,m)$，这时可计算出式(4)的值为1。

考虑实值函数空间$\mathcal{F}:\mathcal{Z} \to \mathbb{R}$。令$Z = \\{ \mathbf{z}_1,\mathbf{z}_2,...,\mathbf{z}_m \\}$，其中$\mathbf{z}_i \in \mathcal{Z}$，将式(4)中的$\mathcal{X}$和$\mathcal{H}$替换为$\mathcal{Z}$和$\mathcal{F}$可得：

**定义12.8** 函数空间$\mathcal{F}$关于$Z$的经验Rademacher复杂度：

$$\hat{R}_Z(\mathcal{F}) = \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{f \in \mathcal{F}} \frac{1}{m} \sum_{i=1}^m \sigma_i f(\mathbf{z}_i) \right] \tag{5}$$

经验Rademacher复杂度衡量了函数空间$\mathcal{F}$与随机噪声在集合$Z$中的相关性。通常我们希望了解函数空间$\mathcal{F}$在$\mathcal{Z}$上关于分布$\mathcal{D}$的相关性，因此，对所有从$\mathcal{D}$独立同分布采样而得的大小为$m$的集合$Z$求期望可得：

**定义12.9** 函数空间$\mathcal{F}$关于$\mathcal{Z}$上分布$\mathcal{D}$的Rademacher复杂度：

$$R_m (\mathcal{F}) = \mathbb{E}_{Z \subseteq \mathcal{Z} : \lvert Z \rvert = m} \left[ \hat{R}_Z (\mathcal{F}) \right] \tag{6}$$

基于Rademacher复杂度可得关于函数空间$\mathcal{F}$的泛化误差界：

**定理12.5** 对实值函数空间$\mathcal{F}:\mathcal{Z} \to [0,1]$，根据分布$\mathcal{D}$从$\mathcal{Z}$中独立同分布采样得到示例集$Z=\\{ \mathbf{z}_1 , \mathbf{z}_2 , ... , \mathbf{z}_m \\},\mathbf{z}_i \in \mathcal{Z},0<\delta <1$，对任意$f \in \mathcal{F}$，以至少$1-\delta$的概率有：

$$\mathbb{E} \left[ f(\mathbf{z}) \right] \leqslant \frac{1}{m} \sum_{i=1}^m f(\mathbf{z}_i) + 2R_m(\mathcal{F}) + \sqrt{\frac{\ln (1 / \delta)}{2m}} \tag{7}$$

$$\mathbb{E} \left[ f(\mathbf{z}) \right] \leqslant \frac{1}{m} \sum_{i=1}^m f(\mathbf{z}_i) + 2\hat{R}_Z(\mathcal{F}) +3 \sqrt{\frac{\ln (2 / \delta)}{2m}} \tag{8}$$

需注意的是，定理12.5中的函数空间$\mathcal{F}$是区间$[0,1]$上的实值函数，因此定理12.5只适用于回归问题。对二分类问题，我们有下面的定理：

**定理12.6** 对假设空间$\mathcal{H}:\mathcal{X} \to \\{ -1,+1 \\}$，根据分布$\mathcal{D}$从$\mathcal{X}$中独立同分布采样得到示例集$D=\\{ \mathbf{x}_1, \mathbf{x}_2, ... , \mathbf{x}_m \\}, \mathbf{x}_i \in \mathcal{X},0<\delta <1$，对任意$h \in \mathcal{H}$，以至少$1-\delta$的概率有：

$$E(h) \leqslant \hat{E}(h) + R_m (\mathcal{H}) + \sqrt{\frac{\ln (1 / \delta)}{2m}} \tag{9}$$

$$E(h) \leqslant \hat{E}(h) + \hat{R}_D(\mathcal{H}) + 3 \sqrt{\frac{\ln (2/ \delta)}{2m}} \tag{10}$$

定理12.6给出了基于Rademacher复杂度的泛化误差界。与定理12.3对比可知，基于VC维的泛化误差界是分布无关、数据独立的，而基于Rademacher复杂度的泛化误差界式(9)与分布$\mathcal{D}$有关，式(10)与数据$D$有关。换言之，基于Rademacher复杂度的泛化误差界依赖于具体学习问题上的数据分布，有点类似于为该学习问题“量身定制”的，因此它通常比基于VC维的泛化误差界更紧一些。

值得一提的是，关于Rademacher复杂度与增长函数，有如下定理：

**定理12.7** 假设空间$\mathcal{H}$的Rademacher复杂度$R_m(\mathcal{H})$与增长函数$\Pi _{\mathcal{H}}(m)$满足：

$$R_m(\mathcal{H}) \leqslant \sqrt{\frac{2\ln \Pi_{\mathcal{H}}(m)}{m}} \tag{11}$$

由式(9)、式(11)和[推论12.2](http://shichaoxin.com/2024/10/14/机器学习基础-第五十三课-计算学习理论-VC维/)可得：

$$E(h) \leqslant \hat{E}(h) + \sqrt{\frac{2d\ln \frac{em}{d}}{m}} + \sqrt{\frac{\ln (1 / \delta)}{2m}} \tag{12}$$

也就是说，我们从Rademacher复杂度和增长函数能推导出基于VC维的泛化误差界。

# 2.对式(1)的解释

这里解释从第一步到第二步的推导，因为前提假设是2分类问题，$y_k \in \\{ -1,+1 \\}$，因此$\mathbb{I}(h(\mathbf{x_i}) \neq y_i) \equiv \frac{1-y_i h(\mathbf{x}_i)}{2}$。这是因为假如$y_i = +1,h(\mathbf{x}_i)=+1$或$y_i = -1,h(\mathbf{x}_i) = -1$，有$\mathbb{I}(h(\mathbf{x}_i)\neq y_i) = 0 = \frac{1-y_i h(\mathbf{x}_i)}{2}$；反之，假如$y_i=-1,h(\mathbf{x}_i)=+1$或$y_i=+1,h(\mathbf{x}_i)=-1$，有$\mathbb{I}(h(\mathbf{x}_i)\neq y_i) = 1 = \frac{1-y_i h(\mathbf{x}_i)}{2}$。

# 3.定理12.5的证明

>对这部分证明不太感兴趣，就没细看，这里贴出原书中的证明，感兴趣的小伙伴们可以看一看。

令：

$$\hat{E}_Z(f) = \frac{1}{m} \sum_{i=1}^m f(\mathbf{z}_i)$$

$$\Phi (Z) = \sup _{f \in \mathcal{F}} \mathbb{E} [f] - \hat{E}_Z(f)$$

同时，令$Z'$为只与$Z$有一个示例不同的训练集，不妨设$\mathbf{z}_m \in Z$和$\mathbf{z}'_m \in Z'$为不同示例，可得：

$$\begin{align} \Phi (Z') - \Phi (Z) &= \left( \sup_{f \in \mathcal{F}} \mathbb{E}[f] - \hat{E}_{Z'}(f) \right) - \left( \sup_{f \in \mathcal{F}} \mathbb{E}[f] - \hat{E}_Z(f) \right) \\& \leqslant \sup_{f \in \mathcal{F}} \hat{E}_Z (f) - \hat{E}_{Z'}(f) \\&= \sup_{f \in \mathcal{F}} \frac{f(\mathbf{z}_m)-f(\mathbf{z}'_m)}{m} \\& \leqslant \frac{1}{m} \end{align}$$

同理可得：

$$\Phi (Z) - \Phi (Z') \leqslant \frac{1}{m}$$

$$\lvert \Phi (Z) - \Phi (Z') \rvert \leqslant \frac{1}{m}$$

根据[McDiarmid不等式](http://shichaoxin.com/2024/08/24/机器学习基础-第五十课-计算学习理论-基础知识/)可知，对任意$\delta \in (0,1)$：

$$\Phi (Z) \leqslant \mathbb{E}_Z [ \Phi (Z) ] + \sqrt{\frac{\ln (1/\delta)}{2m}} \tag{3.1}$$

以至少$1-\delta$的概率成立。下面来估计$\mathbb{E}_Z[\Phi (Z)]$的上界：

$$\begin{align} \mathbb{E}_Z [\Phi (Z)] &= \mathbb{E}_Z \left[ \sup_{f\in \mathcal{F}} \mathbb{E}[f] - \hat{E}_Z(f) \right] \\&= \mathbb{E}_Z \left[ \sup_{f\in \mathcal{F}} \mathbb{E}_{Z'} [ \hat{E}_{Z'}(f) - \hat{E}_Z(f) ] \right] \\& \leqslant \mathbb{E}_{Z,Z'} \left[ \sup_{f\in \mathcal{F}} \hat{E}_{Z'}(f) - \hat{E}_Z(f) \right] \\&= \mathbb{E}_{Z,Z'} \left[ \sup_{f\in \mathcal{F}} \frac{1}{m} \sum_{i=1}^m (f(\mathbf{z}'_i)-f(\mathbf{z}_i)) \right] \\&= \mathbb{E}_{\boldsymbol{\sigma},Z,Z'} \left[ \sup_{f\in \mathcal{F}} \frac{1}{m} \sum_{i=1}^m \sigma_i (f(\mathbf{z}'_i)-f(\mathbf{z}_i)) \right] \\& \leqslant \mathbb{E}_{\boldsymbol{\sigma},Z'} \left[ \sup_{f\in \mathcal{F}} \frac{1}{m} \sum_{i=1}^m \sigma_i f(\mathbf{z}'_i) \right] + \mathbb{E}_{\boldsymbol{\sigma},Z} \left[ \sup_{f \in \mathcal{F}} \frac{1}{m} \sum_{i=1}^m - \sigma_i f(\mathbf{z}_i) \right] \\&= 2 \mathbb{E}_{\boldsymbol{\sigma},Z} \left[ \sup_{f \in \mathcal{F}} \frac{1}{m} \sum_{i=1}^m \sigma_i f(\mathbf{z}_i) \right] \\&= 2 R_m (\mathcal{F}) \end{align}$$

>利用[Jensen不等式](http://shichaoxin.com/2024/08/24/机器学习基础-第五十课-计算学习理论-基础知识/)和上确界函数的凸性。
>
>$\sigma_i$与$-\sigma_i$分布相同。

至此，式(7)得证。由定义12.9可知，改变$Z$中的一个示例对$\hat{R}_Z(\mathcal{F})$的值所造成的改变最多为$1/m$。由[McDiarmid不等式](http://shichaoxin.com/2024/08/24/机器学习基础-第五十课-计算学习理论-基础知识/)可知：

$$R_m(\mathcal{F}) \leqslant \hat{R}_Z(\mathcal{F}) + \sqrt{\frac{\ln (2/ \delta)}{2m}} \tag{3.2}$$

以至少$1-\delta / 2$的概率成立。再由式(3.1)可知：

$$\Phi (Z) \leqslant \mathbb{E}_Z [\Phi (Z)] + \sqrt{\frac{\ln (2/\delta)}{2m}}$$

以至少$1-\delta / 2$的概率成立。于是：

$$\Phi (Z) \leqslant 2 \hat{R}_Z (\mathcal{F}) + 3 \sqrt{\frac{\ln (2 / \delta)}{2m}} \tag{3.3}$$

以至少$1-\delta$的概率成立。至此，式(8)得证。

# 4.定理12.6的证明

>对这部分证明不太感兴趣，就没细看，这里贴出原书中的证明，感兴趣的小伙伴们可以看一看。

对二分类问题的假设空间$\mathcal{H}$，令$\mathcal{Z} = \mathcal{X} \times \\{ -1,+1 \\}$，则$\mathcal{H}$中的假设$h$变形为：

$$f_h(\mathbf{z}) = f_h(\mathbf{x},y) = \mathbb{I}(h(\mathbf{x})\neq y) \tag{4.1}$$

于是就可将值域为$\\{-1,+1 \\}$的假设空间$\mathcal{H}$转化为值域为$[0,1]$的函数空间$\mathcal{F}_{\mathcal{H}} = \\{ f_h : h \in \mathcal{H} \\}$。由定义12.8，有：

$$\begin{align} \hat{R}_Z(\mathcal{F}_\mathcal{H}) &= \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{f_h \in \mathcal{F}_\mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i f_h (\mathbf{x}_i,y_i) \right] \\&= \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i \mathbb{I}(h(\mathbf{x}_i)\neq y_i) \right] \\&= \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i \frac{1-y_i h(\mathbf{x}_i)}{2} \right] \\&= \frac{1}{2} \mathbb{E}_{\boldsymbol{\sigma}} \left[ \frac{1}{m} \sum_{i=1}^m \sigma_i + \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \left( -y_i\sigma_i h(\mathbf{x}_i) \right) \right] \\&= \frac{1}{2} \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \left( -y_i\sigma_i h(\mathbf{x}_i) \right) \right] \\&= \frac{1}{2} \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{h\in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \left( \sigma_i h(\mathbf{x}_i) \right) \right] \\&= \frac{1}{2} \hat{R}_D (\mathcal{H}) \end{align} \tag{4.2}$$

>$-y_i\sigma_i$与$\sigma_i$分布相同。

对式(4.2)求期望后可得：

$$R_m(\mathcal{F}_{\mathcal{H}}) = \frac{1}{2} R_m (\mathcal{H}) \tag{4.3}$$

由定理12.5和式(4.2)~(4.3)，定理12.6得证。

# 5.参考资料

1. [南瓜书PumpkinBook](https://datawhalechina.github.io/pumpkin-book/#/chapter12/chapter12?id=_125-rademacher复杂度)