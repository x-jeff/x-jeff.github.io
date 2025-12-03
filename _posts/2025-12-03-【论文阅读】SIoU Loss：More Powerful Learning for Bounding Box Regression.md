---
layout:     post
title:      【论文阅读】SIoU Loss：More Powerful Learning for Bounding Box Regression
subtitle:   SIoU
date:       2025-12-03
author:     x-jeff
header-img: blogimg/20221107.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

不再详述。

# 2.Methods

SIoU的全称是SCYLLA-IoU。SIoU损失函数包含4部分：

* 角度损失（angle cost）
* 距离损失（distance cost）
* 形状损失（shape cost）
* IoU损失（IoU cost）

## 2.1.Angle cost

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/1.png)

角度损失的作用就是让预测bbox和GT box在某一个方向（X方向或Y方向）上尽可能的接近。如Fig1所示，如果$\alpha \leqslant \frac{\pi}{4}$，那就最小化$\alpha$，此时预测bbox和GT box会在Y方向上接近；如果$\alpha > \frac{\pi}{4}$，那就最小化$\beta$，此时预测bbox和GT box就会在X方向上接近。也就是说，角度损失只能保证在一个方向上的优化。先优化一个方向，这样能使训练过程更稳定，收敛更快。

角度损失的计算公式见下：

$$\Lambda = 1-2*\sin ^2 (\arcsin (x)- \frac{\pi}{4}) \tag{1}$$

其中，

$$x = \frac{c_h}{\sigma}=\sin(\alpha) \tag{2}$$

$$\sigma = \sqrt{(b_{c_x}^{gt}-b_{c_x})^2+(b_{c_y}^{gt}-b_{c_y})^2} \tag{3}$$

$$c_h = \max (b_{c_y}^{gt},b_{c_y}) - \min (b_{c_y}^{gt},b_{c_y}) \tag{4}$$

$$c_w = \max (b_{c_x}^{gt},b_{c_x}) - \min (b_{c_x}^{gt},b_{c_x}) \tag{5}$$

$(b_{c_x},b_{c_y})$为预测bbox的中心点坐标，$(b_{c_x}^{gt},b_{c_y}^{gt})$为GT box的中心点坐标。$c_h,c_w$的定义见式(4)、式(5)和Fig1。

从上述公式定义来看，角度$\alpha$的值（即$\arcsin (x)$），在0到$\frac{\pi}{2}$之间。当$\alpha$为0°或90°时，此时两个box在X方向或Y方向上对齐，此时角度损失最小，为0。当$\alpha$为45°时，此时角度损失最大，为1。

如下图所示，蓝色区域为有效取值区域，角度损失$\Lambda$的取值范围为$[0,1]$，$x$的取值范围也为$[0,1]$，当$x = \frac{\sqrt{2}}{2}$（即$\sin (45°)$）时，角度损失取到最大值1。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/2.png)

## 2.2.Distance cost

距离损失的定义见下：

$$\Delta = \sum_{t=x,y}(1-e^{-\gamma \rho_t}) \tag{6}$$

其中，

$$\rho_x = \left( \frac{b_{c_x}^{gt}-b_{c_x}}{c_w} \right)^2 \tag{7}$$

$$\rho_y = \left( \frac{b_{c_y}^{gt}-b_{c_y}}{c_h} \right)^2 \tag{8}$$

$$\gamma = 2 - \Lambda \tag{9}$$

式(7)、式(8)中的$c_w,c_h$定义见Fig3（注意：和角度损失中用到的$c_w,c_h$定义不同）：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/3.png)

式(9)中的$\Lambda$为角度损失，其定义见式(1)。

当$\alpha$接近0°或90°时，$\Lambda$会变小，而$\Delta$会变大。当$\alpha$接近45°时，$\Lambda$会变大，而$\Delta$会变小。这样做的意义在于，让模型优先优化角度损失，当两个box在$X$方向或$Y$方向对齐之后，再着重优化距离损失（即让两个box在$X$方向或$Y$方向上更靠近），这样能使得训练更加稳定，收敛更快。

## 2.3.Shape cost

形状损失的定义如下：

$$\Omega = \sum_{t=w,h}(1-e^{-\omega_t})^{\theta} \tag{10}$$

其中，

$$\omega_w = \frac{\lvert w-w^{gt} \rvert}{\max (w,w^{gt})} \tag{11}$$

$$\omega_h = \frac{\lvert h-h^{gt} \rvert}{\max (h,h^{gt})} \tag{12}$$

权重$\theta$越小，形状损失的比重越大，但如果把$\theta$设置的过小，比如$\theta = 1$，这可能会导致训练不稳定。作者推荐$\theta$的取值范围在2到6之间，理想值约为4。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/4.png)

最终，SIoU loss可定义为：

$$L_{box} = 1 - IoU + \frac{\Delta + \Omega}{2} \tag{13}$$

其中，

$$IoU = \frac{\lvert B \cap B^{GT}  \rvert}{ \lvert B \cup B^{GT} \rvert} \tag{14}$$

## 2.4.Training

在COCO-train上训练了300个epoch，然后在COCO-val上进行测试。

## 2.5.Simulation Experiment

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/5.png)

如Fig5所示，我们设置了7个GT box，这7个GT box的中心点坐标都是$(10,10)$且面积都是1，7个GT box的长宽比分别为$(1:4, 1:3, 1:2, 1:1, 2:1, 3:1, 4:1)$。接下来，我们以$(10,10)$为圆心，3为半径，均匀采样5000个点。对于这5000个点中的每一个点，都设置49个anchor box，即7种尺度$\times$7种长宽比，7种尺度（即anchor box的面积）分别为$(0.5,0.67,0.75,1,1.33,1.5,2)$，7种长宽比分别为$(1:4, 1:3, 1:2, 1:1, 2:1, 3:1, 4:1)$，因此一共有$5000 \times 7 \times 7$个anchor box。然后每个anchor box都要去拟合各个GT box，因此，一共会有$5000 \times 7 \times 7 \times 7 = 1715000$个回归测试用例。

最终的总误差可用下式计算：

$$E(i)=\sum_{n=0}^{5000} \sum_{t=x,y,w,h} \lvert B_t^n - B_t^{GT_n} \rvert \tag{15}$$

其中，$B^n$表示预测的bbox，$B^{GT_n}$表示对应的GT box，$E(i)$表示第$i$次迭代的误差。注意，在式(15)中，遍历的是每个点，但其实一个点需要考虑49个anchor box，每个anchor box分别对应7个GT box，所以一共涉及$49 \times 7$个回归测试用例。

训练使用[Adam优化器](https://shichaoxin.com/2020/03/19/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E5%8D%81%E4%B9%9D%E8%AF%BE-Adam%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/)搭配[step learning rate scheduler](https://shichaoxin.com/2020/03/23/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%BA%8C%E5%8D%81%E8%AF%BE-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%A1%B0%E5%87%8F/#34%E6%96%B9%E5%BC%8F%E5%9B%9B)，初始学习率设置为0.1。一共训练了100个epoch。

## 2.6.Implementation test

最终的损失函数包含2部分：分类损失和box loss。

$$L = W_{box} L_{box} + W_{cls} L_{cls} \tag{16}$$

其中，$L_{cls}$是[focal loss](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)。$W_{box},W_{cls}$分别为box loss和分类损失的权重。

## 2.7.Results and Discussion

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/6.png)

先看Fig6左图，anchor box（即蓝色框）位于45°方向，GT box位于原点处，如果使用SIoU loss，回归结果在第495次epoch时就已经收敛，但如果使用[CIoU loss](https://shichaoxin.com/2025/12/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Distance-IoU-Loss-Faster-and-Better-Learning-for-Bounding-Box-Regression/#43complete-iou-loss)，回归结果在第1000次epoch时依旧没有收敛。Fig6右图是一种相对简单的情况，因为anchor box和GT box在Y方向是对齐的，此时SIoU loss和[CIoU loss](https://shichaoxin.com/2025/12/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Distance-IoU-Loss-Faster-and-Better-Learning-for-Bounding-Box-Regression/#43complete-iou-loss)都能很好的收敛，但明显SIoU loss收敛的更快，仅用了119个epoch。

Fig7展示了5000个点的回归误差，很明显，SIoU loss的回归误差更小：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/7.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/8.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/9.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/10.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/11.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/SIoU/12.png)

# 3.Conclusion

不再详述。

# 4.原文链接

👽[SIoU Loss：More Powerful Learning for Bounding Box Regression](https://github.com/x-jeff/AI_Papers/blob/master/2025/SIoU%20Loss%EF%BC%9AMore%20Powerful%20Learning%20for%20Bounding%20Box%20Regression.pdf)