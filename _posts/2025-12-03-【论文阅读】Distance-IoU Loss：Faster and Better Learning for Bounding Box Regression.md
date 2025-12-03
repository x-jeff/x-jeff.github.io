---
layout:     post
title:      【论文阅读】Distance-IoU Loss：Faster and Better Learning for Bounding Box Regression
subtitle:   DIoU，CIoU
date:       2025-12-03
author:     x-jeff
header-img: blogimg/20221122.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

IoU的计算见下：

$$IoU = \frac{\lvert B \cap B^{gt} \rvert}{\lvert B \cup B^{gt} \rvert} \tag{1}$$

其中，$B^{gt} = (x^{gt},y^{gt},w^{gt},h^{gt})$为GT box，$B=(x,y,w,h)$为预测的box。IoU loss被定义为：

$$\mathcal{L}_{IoU} = 1 - \frac{\lvert B \cap B^{gt} \rvert}{\lvert B \cup B^{gt} \rvert} \tag{2}$$

但是，IoU loss仅在box之间有重叠时生效，对于不重叠的案例，并不能有效的提供梯度信息。因此，[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)被提出来用于解决这些问题：

$$\mathcal{L}_{GIoU} = 1 - IoU + \frac{\lvert C - B \cup B^{gt} \rvert}{\lvert C \rvert} \tag{3}$$

其中，$C$是可以覆盖$B$和$B^{gt}$的最小box。

但[GIoU](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)也存在局限性。如Fig1所示，绿色框为GT box，黑色框为anchor box，蓝色框为使用[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)预测得到的box，红色框为使用DIoU loss预测得到的box。可以看到，[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)倾向于先增大预测box的尺寸，使其首先与GT box产生重叠，然后式(3)中的IoU部分才开始起作用，以最大化box之间的重叠区域，这导致[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)整体收敛非常慢。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/1.png)

在Fig2中，绿色框为GT box，红色框为预测box。可以看到，如果预测box的中心点越接近GT box的中心点，DIoU loss就越小，但是IoU loss和[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)却没有变化。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/2.png)

# 2.Related Work

不再详述。

# 3.Analysis to IoU and GIoU Losses

## 3.1.Simulation Experiment

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/3.png)

如Fig3左图所示，在模拟实验中，设置7个GT box，这7个GT box的面积均为1且中心点都位于$(10,10)$处，但这7个GT box的长宽比各不相同，分别为$(1:4, \  1:3, \  1:2,\  1:1, \  2:1,\  3:1,\   4:1)$。然后以$(10,10)$为圆心，3为半径，均匀取5000个点，在每个点上设置49个anchor box（7种尺度$\times$7种长宽比），anchor box所用的7种尺度分别为$(0.5,\  0.67,\  0.75,\  1,\  1.33,\   1.5,\  2)$，anchor box所用的7种长宽比分别为$(1:4, \  1:3, \  1:2,\  1:1, \  2:1,\  3:1,\   4:1)$。因此，一共会有$5000 \times 7 \times 7$个anchor box，每个anchor box都需要分别回归到每个GT box，所以一共会有$7\times 7 \times 7 \times 5000 = 1715000$个回归用例。

模拟实验的流程如下所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/4.png)

$\mathbb{M}$表示的是所有的1715000个anchor box的集合，$\mathbb{M}^{gt}$表示的是7个GT box。最终输出的$E(t,n)$表示每个点（一共5000个点）在不同迭代次数时的回归误差。

$$B_i^t = B_i^{t-1} + \eta (2-IoU_i^{t-1}) \nabla B_i^{t-1} \tag{4}$$

式(4)表示的是使用梯度下降法计算当前预测box的过程。其中，$B_i^t$表示在第$t$次迭代时的预测框，$\nabla B_i^{t-1}$表示损失函数$\mathcal{L}$在第$t-1$次迭代时对预测框$B_i^t$的梯度，$\eta$是步长，此外，梯度被额外乘以$2-IoU_i^{t-1}$以加速收敛。

在Fig3右图中，我们设置最大迭代次数$T=200$，纵轴表示在该迭代次数时，所有5000个点的回归误差总和，即$\sum_n \mathbf{E}(t,n)$。可以看到，在迭代到第200次时，DIoU和CIoU的回归误差远远小于IoU和[GIoU](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)。

## 3.2.Limitations of IoU and GIoU Losses

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/5.png)

在Fig4中，我们可视化了在迭代次数$T$结束后，5000个散点对应的最终回归误差。从Fig4(a)可以很容易看出，IoU loss只在预测box和GT box存在重叠的情况下才有效。对于没有重叠的情况，由于梯度$\nabla B$一直是0，因此anchor box不会发生任何更新。

[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)可以更好的缓解没有重叠的情况。从Fig4(b)可以看到，[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)显著扩大了其有效区域，即能够有效回归的区域。但在水平或垂直方向对齐的情况下，误差仍然可能非常大。这是因为[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)中的惩罚项试图最小化$\lvert C - A \cup B \rvert$，当$A,B$具有包含关系时，$\lvert C - A \cup B \rvert$往往很小甚至为0，此时[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)几乎退化为IoU loss。只要使用足够的迭代次数和合理的学习率，[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)最终可以收敛到一个较好的解，但其收敛速度非常慢。

在实际目标检测工作流中，IoU和[GIoU](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)都无法保证回归精度。

因此，自然而然的引出两个问题：

1. 是否可以直接最小化预测box和GT box之间的归一化距离，从而获得更快的收敛速度？
2. 在有重叠甚至互相包含的情况下，如何使回归过程更加准确、更加快速？

# 4.The Proposed Method

通常，基于IoU的损失函数可定义为：

$$\mathcal{L} = 1-IoU+\mathcal{R}(B,B^{gt}) \tag{5}$$

其中，$\mathcal{R}(B,B^{gt})$是针对预测box和GT box的惩罚项。我们的目标就是设计出合理的惩罚项。

## 4.1.Distance-IoU Loss

为了解决第3.2部分提出的第一个问题，我们提出直接最小化两个box中心点之间的归一化距离，其惩罚项可定义为：

$$\mathcal{R}_{DIoU} = \frac{\rho ^2 (\mathbf{b},\mathbf{b}^{gt})}{c^2} \tag{6}$$

其中，$\mathbf{b}$和$\mathbf{b}^{gt}$分别表示预测box和GT box的中心点，$\rho (\cdot)$表示欧氏距离，$c$表示最小外接矩形的对角线长度。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/6.png)

于是，DIoU loss可定义为：

$$\mathcal{L}_{DIoU} = 1-IoU + \frac{\rho ^2 (\mathbf{b},\mathbf{b}^{gt})}{c^2} \tag{7}$$

## 4.2.Comparison with IoU and GIoU losses

DIoU loss继承了IoU loss和[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)的一些性质：

1. DIoU loss仍然具有尺度不变性。
2. 和[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)类似，DIoU loss在预测box和GT box不重叠时，也能为预测box提供有效的移动方向。
3. 当预测box和GT box完美匹配时，有$\mathcal{L}\_{IoU}=\mathcal{L}\_{GIoU}=\mathcal{L}\_{DIoU}=0$。当预测box和GT box相距较远时，有$\mathcal{L}\_{GIoU}=\mathcal{L}\_{DIoU} \to 2$。

此外，相较于IoU loss和[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)，DIoU loss具有多项优势：

1. 如Fig1和Fig3所示，DIoU loss直接最小化两个box之间的中心点距离，因此收敛更快。
2. 如Fig4所示，当两个box存在包含关系，或者水平/垂直方向对齐的情况下，DIoU loss依然可以快速回归，而此时[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)几乎退化为IoU loss。

## 4.3.Complete IoU Loss

接下来我们来回答第3.2部分提出的第二个问题。一个好的回归损失应同时考虑三个重要的几何因素：重叠区域、中心点距离、长宽比。IoU loss和[GIoU loss](https://shichaoxin.com/2025/11/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generalized-Intersection-over-Union-A-Metric-and-A-Loss-for-Bounding-Box-Regression/)只考虑了重叠区域，DIoU loss考虑了重叠区域和中心点距离。因此，基于DIoU loss，我们提出了CIoU loss，同时考虑以上3个重要因素，其惩罚项的定义见下：

$$\mathcal{R}_{CIoU} = \frac{\rho^2 (\mathbf{b},\mathbf{b}^{gt})}{c^2} + \alpha v \tag{8}$$

其中，$\alpha$是一个正的权衡参数，$v$用于衡量长宽比的一致性，定义为：

$$v = \frac{4}{\pi^2} (\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h})^2 \tag{9}$$

CIoU loss可定义为：

$$\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2 (\mathbf{b},\mathbf{b}^{gt})}{c^2} + \alpha v \tag{10}$$

权衡参数$\alpha$可定义为：

$$\alpha = \frac{v}{(1-IoU)+v} \tag{11}$$

梯度的计算如下：

$$\frac{\partial v}{\partial w} =\frac{8}{\pi^2} (\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h}) \times \frac{h}{w^2+h^2} \\ \frac{\partial v}{\partial h} = -\frac{8}{\pi^2} (\arctan \frac{w^{gt}}{h^{gt}}-\arctan \frac{w}{h}) \times \frac{w}{w^2 + h^2} \tag{12}$$

当$h,w$在$[0,1]$范围内时，项$w^2+h^2$会非常小，容易导致梯度爆炸。因此，在实际实现时，为了保持稳定收敛，我们将$\frac{1}{w^2+h^2}$替换为1。

## 4.4.Non-Maximum Suppression using DIoU

对于得分最高的预测框$\mathcal{M}$，DIoU-NMS可定义为：

$$s_i = \begin{cases} s_i, &  IoU-\mathcal{R}_{DIoU}(\mathcal{M},B_i) < \varepsilon, \\ 0, & IoU - \mathcal{R}_{DIoU}(\mathcal{M},B_i) \geqslant \varepsilon, \end{cases} \tag{13}$$

其中，$s_i$为分类分数，$\varepsilon$为NMS阈值。当框$B_i$和$\mathcal{M}$的重叠面积很大（即$IoU$很大），两个框的中心点距离也很接近（即$\mathcal{R}_{DIoU}(\mathcal{M},B_i)$很小），此时通常会满足式(13)中的第2个式子，那么框$B_i$就会被移除。

# 5.Experimental Results

## 5.1.YOLO v3 on PASCAL VOC

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/7.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/8.png)

## 5.2.SSD on PASCAL VOC

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/9.png)

## 5.3.Faster R-CNN on MS COCO

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/10.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/11.png)

## 5.4.Discussion on DIoU-NMS

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/12.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/DIoU/13.png)

# 6.Conclusion

不再赘述。

# 7.原文链接

👽[Distance-IoU Loss：Faster and Better Learning for Bounding Box Regression](https://github.com/x-jeff/AI_Papers/blob/master/2025/Distance-IoU%20Loss%EF%BC%9AFaster%20and%20Better%20Learning%20for%20Bounding%20Box%20Regression.pdf)