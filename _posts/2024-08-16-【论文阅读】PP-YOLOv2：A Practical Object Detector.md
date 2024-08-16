---
layout:     post
title:      【论文阅读】PP-YOLOv2：A Practical Object Detector
subtitle:   PP-YOLOv2
date:       2024-08-16
author:     x-jeff
header-img: blogimg/20220601.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

不再详述。

# 2.Revisit PP-YOLO

baseline模型的实现细节见下。

👉**Pre-Processing.**

假设有两个样本$(x_i,y_i)$和$(x_j,y_j)$，则[MixUp](https://shichaoxin.com/2024/01/14/YOLO%E7%B3%BB%E5%88%97-YOLOv5/#3data-augmentation-techniques)生成的新样本$(\tilde{x},\tilde{y})$表示为：

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$

$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

其中，$x_i,x_j$是输入样本（通常是图像），$y_i,y_j$是输入样本对应的标签，$\lambda$是一个在$[0,1]$区间的随机权重系数，通常从Beta分布中随机采样，即$\lambda \sim Beta(\alpha,\beta)$，$\alpha,\beta$为超参数。

在这里，我们设$\alpha=\beta=1.5$。[MixUp](https://shichaoxin.com/2024/01/14/YOLO%E7%B3%BB%E5%88%97-YOLOv5/#3data-augmentation-techniques)之后，我们逐个施加以下数据扩展：RandomColorDistortion、RandomExpand、RandCrop、RandomFlip，施加的概率都是0.5。接着是RGB的通道归一化，即RGB三个通道分别减去0.485、0.456、0.406，再分别除以0.229、0.224、0.225。最后，输入图像被resize到下列尺寸：$[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]$。

👉**Baseline Model.**

baseline模型使用[PP-YOLO](https://shichaoxin.com/2024/08/13/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-PP-YOLO-An-Effective-and-Efficient-Implementation-of-Object-Detector/)。

👉**Training Schedule.**

在COCO train2017上，使用SGD训练了500K次迭代，minibatch size=96，用了8块GPU。在前4K次迭代中，学习率从0线性增长为0.005，然后在第400K和第450K次迭代时，学习率除以10。weight decay=0.0005，momentum=0.9。为了使训练稳定，使用了gradient clipping。

# 3.Selection of Refinements

👉**Path Aggregation Network.**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/PPYOLOv2/1.png)

如Fig2所示，PP-YOLOv2将detection neck中的[FPN](https://shichaoxin.com/2023/12/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Feature-Pyramid-Networks-for-Object-Detection/)替换为了[PAN](https://shichaoxin.com/2023/12/28/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Path-Aggregation-Network-for-Instance-Segmentation/)。

👉**Mish Activation Function.**

将detection neck中的激活函数替换为[Mish激活函数](https://shichaoxin.com/2024/01/04/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/#34yolov4)。

👉**Larger Input Size.**

增大输入尺寸可以提高性能，但也会占用更多内存。为了解决这个问题，我们降低了batch size。从每个GPU处理24张图像降低到每个GPU只处理12张图像，将最大输入尺寸从608提高到了768。输入尺寸的取值：$[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768]$。

👉**IoU Aware Branch.**

使用如下IoU aware loss：

$$loss = -t * \log (\sigma(p))-(1-t)*\log (1-\sigma(p)) \tag{1}$$

其中，$t$是anchor和其对应的GT box之间的IoU，$p$是IoU aware分支的原始输出，$\sigma(\cdot)$是sigmoid函数。只有阳性样本才会计算IoU aware loss。

# 4.Experiments

## 4.1.Dataset

训练集为COCO train2017（包含118k张图像，共80个类别），在COCO minival（包含5k张图像）上进行评估。评估指标为mAP。

## 4.2.Ablation Studies

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/PPYOLOv2/2.png)

## 4.3.Comparison With Other State-of-the-Art Detectors

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/PPYOLOv2/3.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/PPYOLOv2/4.png)

# 5.Things We Tried That Didn’t Work

[PP-YOLO](https://shichaoxin.com/2024/08/13/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-PP-YOLO-An-Effective-and-Efficient-Implementation-of-Object-Detector/)在COCO train2017数据集上，使用8块V100 GPU，训练了约80个小时，为了节省时间，在消融实验中，我们使用COCO minitrain作为训练集。COCO minitrain是COCO train2017的子集，包含25K张图像。在COCO minitrain上，一共训练了90K次迭代。在第60k次迭代时，将学习率除以10。其他设置和在COCO train2017上的训练是一样的。

在开发PP-YOLOv2的过程中，我们尝试了很多方法。有些方法在COCO minitrain上有效，但是在COCO train2017上却降低了性能。由于这种不一致，有人可能会怀疑在COCO minitrain上的实验结果。我们使用COCO minitrain的原因是想要寻求一些通用性的改进，使其在不同规模的数据集上都有用。这里，我们列出了一些失败的方法。

👉**[Cosine Learning Rate Decay](https://shichaoxin.com/2024/07/10/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks/#51cosine-learning-rate-decay).**

[Cosine Learning Rate Decay](https://shichaoxin.com/2024/07/10/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks/#51cosine-learning-rate-decay)在COCO minitrain上取得了更好的性能，但是在COCO train2017上却没有带来正面的影响。

👉**Backbone Parameter Freezing.**

在ImageNet上预训练好之后，在下游任务上fine-tuning时，冻结前两个stage的参数是一个常见的操作。这一策略在COCO minitrain上带来了1mAP的提升，但在COCO train2017上却导致mAP下降了0.8%。

👉**[SiLU](https://shichaoxin.com/2022/04/09/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-GAUSSIAN-ERROR-LINEAR-UNITS-(GELUS)/) Activation Function.**

我们尝试将detection neck中的[Mish激活函数](https://shichaoxin.com/2024/01/04/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection/#34yolov4)替换为[SiLU激活函数](https://shichaoxin.com/2022/04/09/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-GAUSSIAN-ERROR-LINEAR-UNITS-(GELUS)/)。这在COCO minitrain上带来了0.3%的mAP提升，却在COCO train2017上导致mAP下降了0.5%。

# 6.Conclusions

不再赘述。

# 7.原文链接

👽[PP-YOLOv2：A Practical Object Detector](https://github.com/x-jeff/AI_Papers/blob/master/2024/PP-YOLOv2：A%20Practical%20Object%20Detector.pdf)