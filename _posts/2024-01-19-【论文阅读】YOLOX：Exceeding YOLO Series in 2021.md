---
layout:     post
title:      【论文阅读】YOLOX：Exceeding YOLO Series in 2021
subtitle:   YOLOX
date:       2024-01-19
author:     x-jeff
header-img: blogimg/20200607.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

>源码：[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)。

随着目标检测的发展，YOLO系列（[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)、[YOLOv2](http://shichaoxin.com/2022/06/01/论文阅读-YOLO9000-Better,-Faster,-Stronger/)、[YOLOv3](http://shichaoxin.com/2022/06/29/论文阅读-YOLOv3-An-Incremental-Improvement/)、[YOLOv4](http://shichaoxin.com/2024/01/04/论文阅读-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/)、[YOLOv5](http://shichaoxin.com/2024/01/14/YOLO系列-YOLOv5/)）始终追求速度和精度之间的最佳平衡。目前，[YOLOv5](http://shichaoxin.com/2024/01/14/YOLO系列-YOLOv5/)具有最优的平衡性能，在COCO上以13.7ms的速度达到了48.2%的AP（使用[YOLOv5-L](http://shichaoxin.com/2024/01/14/YOLO系列-YOLOv5/)模型，输入为$640 \times 640$，推理精度为FP16，batch=1，使用V100 GPU）。

过去两年的研究大多集中在anchor-free的检测器上，但[YOLOv4](http://shichaoxin.com/2024/01/04/论文阅读-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/)和[YOLOv5](http://shichaoxin.com/2024/01/14/YOLO系列-YOLOv5/)都是anchor-based的检测器，并且可能存在优化过度的问题。因此，我们基于[YOLOv3](http://shichaoxin.com/2022/06/29/论文阅读-YOLOv3-An-Incremental-Improvement/)进行修改。[YOLOv3](http://shichaoxin.com/2022/06/29/论文阅读-YOLOv3-An-Incremental-Improvement/)的框架图：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/1.png)

作者在[YOLOv3](http://shichaoxin.com/2022/06/29/论文阅读-YOLOv3-An-Incremental-Improvement/)的基础上添加了[SPP结构](http://shichaoxin.com/2022/02/22/论文阅读-Spatial-Pyramid-Pooling-in-Deep-Convolutional-Networks-for-Visual-Recognition/)作为基础默认模型，称为YOLOv3-SPP：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/2.png)

YOLOX的性能见Fig1：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/3.png)

值得一提的是，我们使用单个YOLOX-L模型赢得了Streaming Perception Challenge（Workshop on Autonomous Driving at CVPR 2021）的第一名。

# 2.YOLOX

## 2.1.YOLOX-DarkNet53

👉**Implementation details**

从baseline到final model，我们的训练设置基本一致。我们在COCO train2017上训练了300个epoch，其中5个epoch用于warmup。训练使用SGD。学习率设置$lr \times \frac{\text{BatchSize}}{64}$，其中$lr=0.01$，使用[cosine lr schedule](http://shichaoxin.com/2024/01/04/论文阅读-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/#34yolov4)。weight decay=0.0005，SGD momentum=0.9。使用8块GPU，batch size=128。输入大小从448到832，以32的步长均匀采样。FPS和latency的测试都基于FP16精度，batch=1和单个的Tesla V100。

>latency（延迟）通常指的是模型推理和后处理所需要的时间。

👉**YOLOv3 baseline**

使用YOLOv3-SPP作为baseline。和[原始实现](http://shichaoxin.com/2022/06/29/论文阅读-YOLOv3-An-Incremental-Improvement/)相比，我们修改了一些训练策略：

* [EMA](http://shichaoxin.com/2020/02/25/深度学习基础-第十六课-指数加权平均/)权重更新
* [cosine lr schedule](http://shichaoxin.com/2024/01/04/论文阅读-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/#34yolov4)
* 使用IoU Loss训练reg分支，使用BCE Loss训练cls分支和obj分支。

此外，关于data augmentation，我们只使用了RandomHorizontalFlip、ColorJitter和multi-scale，放弃了RandomResizedCrop，因为我们发现RandomResizedCrop和mosaic augmentation有点重复了。如表2所示，我们改进的baseline在COCO val上取得了38.5%的AP。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/4.png)

👉**Decoupled head**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/5.png)

如Fig2所示，在YOLOv3-v5中，每一个预测分支（来自不同金字塔层级）都是使用一个coupled head来一起预测出cls，reg和obj。而在YOLOX中，每个预测分支（来自不同金字塔层级）使用decoupled head，一个子分支用于预测cls，另一个子分支用于预测reg和obj。下面放一张更直观的解释图：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/6.png)

更多细节：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/7.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/8.png)

我们的两个分析实验表明，coupled head可能会损害性能。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/9.png)

1. 如Fig3所示，用decoupled head替换YOLO head大大提高了收敛速度。
2. decoupled head对YOLO的端到端版本至关重要。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/10.png)

这里的端到端版本指的是没有后处理（即没有NMS）。从表1中可以看出，decoupled head可以使端到端版本性能更高（性能损失更少），并且相比coupled head，性能也更高。decoupled head模型的推理性能见表2。

👉**Strong data augmentation**

使用了[Mosaic](http://shichaoxin.com/2024/01/04/论文阅读-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/#34yolov4)和[MixUp](http://shichaoxin.com/2024/01/04/论文阅读-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/#34yolov4)。在使用了strong data augmentation之后，我们发现在ImageNet上预训练的作用不大了，因此我们的模型都是从头开始训练的。

👉**Anchor-free**

YOLOv3-v5都是anchor-based pipeline。但是，anchor机制存在一些问题。首先，为了实现最佳检测性能，需要在训练前通过聚类分析以确定一组最优的anchor。这些anchor是基于特定领域（特定数据集）的，泛化性不好。其次，anchor机制增加了检测头的复杂性，以及每个图像的预测数量。

最近的一些研究表明，anchor-free模型和anchor-based模型的表现不相上下。

在anchor-based的方法中，以YOLOv3-SPP为例：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/11.png)

如果输入图像大小为$416 \times 416$，则网络最后的三个feature map的大小为$13 \times 13, 26 \times 26, 52 \times 52$。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/12.png)

黄色框为小狗的GT，GT的中心点落在红色cell内，该cell对应的anchor用蓝色框表示。每个cell都有3个anchor。如果我们使用COCO数据集，共有80个类别，则对于每个anchor，预测结果会有85个值：bounding box的位置（4个值）、obj（前景或背景，1个值）、类别（80个值）。因此会产生$3 \times (13 \times 13+ 26 \times 26+ 52 \times 52) \times 85 = 904995$个预测结果。如果将输入从$416 \times 416$变为$640 \times 640$，最后3个feature map大小为$20 \times 20, 40 \times 40, 80 \times 80$。则会产生$3 \times (20 \times 20+ 40 \times 40+ 80 \times 80) \times 85 = 2142 000$个预测结果。

而YOLOX所采用的anchor free的方式：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/13.png)

当输入为$640 \times 640$时，最终输出得到的不是feature map，而是特征向量，大小为$85 \times 8400$，相比之前anchor based方式，少了$\frac{2}{3}$的参数量。在前面anchor based方式中，feature map中的每个cell都有3个大小不一的anchor box，在YOLOX中同样也有类似的机制，其把下采样的大小信息引入进来：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/14.png)

8400表示cell的数量，每个cell对应一种尺度的anchor，这个anchor的大小取决于下采样：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/15.png)

这8400个cell中，有400个（来自$20 \times 20$）cell对应的anchor大小为$32 \times 32$，有1600个（来自$40 \times 40$）cell对应的anchor大小为$16 \times 16$，有6400个（来自$80 \times 80$）cell对应的anchor大小为$8 \times 8$。

>个人理解：因为YOLOX的anchor都是正方形，所以对于瘦长形的目标，比如行人，检测效果可能不如[YOLOv5](http://shichaoxin.com/2024/01/14/YOLO系列-YOLOv5/)。

👉**Multi positives**

如果和[YOLOv3](http://shichaoxin.com/2022/06/29/论文阅读-YOLOv3-An-Incremental-Improvement/)一样，针对上述anchor-free版本的模型，只把GT中心点所在cell的anchor视为正样本，那么就会忽视周边一些高质量的预测，这些预测其实也是有用的，因此我们将GT中心点所在cell及其$3 \times 3$范围内cell的所有anchor都视为正样本（这一技术在[FCOS](http://shichaoxin.com/2024/08/20/论文阅读-FCOS-Fully-Convolutional-One-Stage-Object-Detection/)中被称为"center sampling"）。如表2所示，这一修改带来了性能的提升。

>$3 \times 3$这个范围对于不同目标可能是不同的，比如对于有的目标，范围是$5 \times 5$。

👉**SimOTA**

OTA（Optimal Transport Assignment）是旷视科技提出的一种动态样本匹配算法（YOLOX也是旷视科技同年提出的）。所谓的样本匹配就是在训练前，我们需要将样本标记好标签，比如有些anchor box被标记为正样本，有些anchor box被标记为负样本。而动态样本匹配就是在训练过程中，样本的标签是动态变化的，比如同一个anchor box在上一轮训练中被标记为正样本，在下一轮训练中就有可能被标记为负样本。

>OTA论文：Zheng Ge, Songtao Liu, Zeming Li, Osamu Yoshie, and Jian Sun. Ota: Optimal transport assignment for object detection. In CVPR, 2021.。

在我们的认知中，样本匹配有4个因素十分重要（以下来自旷视科技在知乎上的回答，详见[参考资料3](https://www.zhihu.com/question/473350307)）：

1. loss/quality/prediction aware：基于网络自身的预测来计算anchor box或者anchor point与gt的匹配关系，充分考虑到了不同结构/复杂度的模型可能会有不同行为，是一种真正的dynamic样本匹配。而loss aware后续也被发现对于DeTR和DeFCN这类端到端检测器至关重要。与之相对的，基于IoU阈值/in Grid (YOLOv1)/in Box or Center ([FCOS](http://shichaoxin.com/2024/08/20/论文阅读-FCOS-Fully-Convolutional-One-Stage-Object-Detection/))都属于依赖人为定义的几何先验做样本匹配，目前来看都属于次优方案。
2. center prior：考虑到感受野的问题，以及大部分场景下，目标的质心都与目标的几何中心有一定的联系，将正样本限定在目标中心的一定区域内做loss/quality aware样本匹配能很好地解决收敛不稳定的问题。
3. 不同目标设定不同的正样本数量（dynamic k）：我们不可能为同一场景下的西瓜和蚂蚁分配同样的正样本数，如果真是那样，那要么蚂蚁有很多低质量的正样本，要么西瓜仅仅只有一两个正样本。dynamic k的关键在于如何确定k，有些方法通过其他方式间接实现了动态k，比如[ATSS](http://shichaoxin.com/2024/09/25/论文阅读-Bridging-the-Gap-Between-Anchor-based-and-Anchor-free-Detection-via-Adaptive-Training-Sample-Selection/)、PAA，甚至[RetinaNet](http://shichaoxin.com/2024/02/22/论文阅读-Focal-Loss-for-Dense-Object-Detection/)，同时，k的估计依然可以是prediction aware的，我们具体的做法是首先计算每个目标最接近的10个预测，然后把这10个预测与gt的IoU加起来求得最终的k，很简单有效，对10这个数字也不是很敏感，在5～15调整几乎没有影响。
4. 全局信息：有些anchor box/point处于正样本之间的交界处，或者正负样本之间的交界处，这类anchor box/point的正负划分，甚至若为正，该是谁的正样本，都应充分考虑全局信息。

OTA就是满足上述4点的，一个好的样本匹配策略。但是OTA最大的问题是会增加约20%-25%的额外训练时间，这对于动辄300 epoch的COCO训练来说是有些吃不消的，因此我们去掉了OTA里的最优方案求解过程（即去掉了Sinkhorn-Knopp算法），保留上面4点的前3点，简而言之：loss aware dynamic top k，我们将其称为SimOTA（Simplified OTA）。接下来我们来详细介绍下SimOTA算法。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/16.png)

如上图所示，假设我们的图像大小为$20 \times 20$，有3个GT（绿色框）。对于每个GT，按照其中心，取一个边长为$n$的`fixed center area`，如上图所示，我们用蓝色框表示这个区域，我们取$n=5$。一共有94个anchor point落在了GT和`fixed center area`的并集中，我们计算这些anchor point和每一个GT的IoU，得到如下IoU矩阵：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/17.png)

类似的，计算这些anchor point和每一个GT的cost，得到如下cost矩阵：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/18.png)

其中，GT $g_i$和anchor point $p_j$（即预测得到的bounding box）之间的cost的计算方式为：

$$c_{ij} = L_{ij}^{cls} + \lambda L_{ij}^{reg} \tag{1}$$

其中，$L_{ij}^{cls}$是$g_i$和$p_j$之间的分类loss，而$L_{ij}^{reg}$是$g_i$和$p_j$之间的回归loss。

接下来说下dynamic k的确定。在IoU矩阵中，对于每一个GT，我们取最大的10个IoU求和，然后取整得到dynamic k。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/19.png)

每个GT，根据自己的dynamic k，找到cost最小的k个anchor point作为自己的正样本：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/20.png)

一个anchor point只能作为一个GT的正样本，但是上图中，A4同时是GT0和GT1的正样本，此时我们会匹配cost更小的一对，所以A4最终和GT1匹配，最终匹配结果见下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/21.png)

SimOTA对性能的提升见表2。

👉**End-to-end YOLO**

我们按照论文“Qiang Zhou, Chaohui Yu, Chunhua Shen, Zhibin Wang, and Hao Li. Object detection made simpler by eliminating heuristic nms. arXiv preprint arXiv:2101.11782, 2021.”，添加了两个额外的卷积层，一对一的标签分配和停止梯度。这些修改使得检测器能够以端到端的方式执行，但这略微降低了性能和推理速度，见表2。因此，我们将其作为一个可选模块，在最终模型中并没有使用。

## 2.2.Other Backbones

除了DarkNet53，我们还测试了其他backbone，YOLOX框架的性能都比对应的counterparts要好。

👉**Modified CSPNet in YOLOv5**

为了公平比较，我们采用了[YOLOv5](http://shichaoxin.com/2024/01/14/YOLO系列-YOLOv5/)的backbone，包含修改后的CSPNet、SiLU激活函数和PAN head。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/22.png)

👉**Tiny and Nano detectors**

我们进一步将我们的模型缩小为YOLOX-Tiny，并和[YOLOv4-Tiny](https://shichaoxin.com/2025/07/15/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Scaled-YOLOv4-Scaling-Cross-Stage-Partial-Network/)进行了比较。对于移动设备，我们采用depth-wise卷积来构建YOLOX-Nano模型。比较结果见表4。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/23.png)

👉**Model size and data augmentation**

在我们的实验中，几乎所有模型都使用相同的学习策略和优化参数，如第2.1部分所示。但是我们发现不同大小的模型，其适合的数据增强也不同。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/24.png)

YOLOX-S、YOLOX-Tiny和YOLOX-Nano都适用于小模型的数据增强方式。

# 3.Comparison with the SOTA

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOX/25.png)

还有一些高性能的YOLO系列大模型，比如Scale-YOLOv4和YOLOv5-P6。还有最近基于Transformer的检测器，把精度提高到了~60 AP的SOTA水平。但由于时间和资源限制，我们没有和这些方法比较。

# 4.1st Place on Streaming Perception Challenge (WAD at CVPR 2021)

不再详述。

# 5.Conclusion

不再详述。

# 6.原文链接

👽[YOLOX：Exceeding YOLO Series in 2021](https://github.com/x-jeff/AI_Papers/blob/master/YOLOX：Exceeding%20YOLO%20Series%20in%202021.pdf)

# 7.参考资料

1. [深入浅出Yolo系列之Yolox核心基础完整讲解](https://zhuanlan.zhihu.com/p/397993315)
2. [YOLOX——SimOTA图文详解](https://zhuanlan.zhihu.com/p/609370771)
3. [如何评价旷视开源的YOLOX，效果超过YOLOv5?](https://www.zhihu.com/question/473350307)