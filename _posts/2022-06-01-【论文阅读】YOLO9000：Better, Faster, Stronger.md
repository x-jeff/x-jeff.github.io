---
layout:     post
title:      【论文阅读】YOLO9000：Better, Faster, Stronger
subtitle:   YOLOv2，YOLO9000
date:       2022-06-01
author:     x-jeff
header-img: blogimg/20220601.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

我们希望目标检测可以更快，更准，识别更多不同类别的object。但是目前多数的目标检测算法只能识别一小部分类别的object。

此外，不同于图像分类数据集（数据集规模通常比较大），目标检测数据集的规模通常比较小，因为标注object的成本过高。

我们提出一种新的方法以使用图像分类数据来完善我们的目标检测系统。

我们还提出了一种联合训练算法，允许我们在目标检测和图像分类数据集上训练我们的目标检测器。我们的方法通过目标检测数据来学习对object的精准定位，通过图像分类数据来增加算法的鲁棒性。

基于此，我们训练了YOLO9000，一个real-time的目标检测器，可以检测超过9000个不同类别的object。首先，我们对[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)进行了优化，得到YOLOv2。然后通过我们的数据集结合方法以及联合训练算法将模型训练至支持超过9000个类别（来自ImageNet和COCO）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/1.png)

代码和pre-trained的模型见：[http://pjreddie.com/yolo9000/](http://pjreddie.com/yolo9000/)。

# 2.Better

相比于SOTA的目标检测算法，[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)仍有许多缺点。在[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)提供的和[Fast R-CNN](http://shichaoxin.com/2022/03/07/论文阅读-Fast-R-CNN/)的error analysis中，[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)的localization error很高。此外，和region proposal-based methods相比，[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)的recall很低。因此，我们的优化方向是在保住[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)的分类准确度的基础上，进一步提升recall，降低localization error。

我们并不想通过使用更大、更深的网络来增加性能，因为这样会损失速度。最后我们优化的过程见表2：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/2.png)

接下来我们来逐个看下这些优化策略。

**[Batch Normalization](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)**

对[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)中的所有卷积层添加[BatchNorm](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)，mAP提升了2%左右。使用了[BatchNorm](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)之后，我们即使移除[dropout](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#5dropout正则化)，模型也不会过拟合。

**High Resolution Classifier**

所有SOTA的检测方法都在ImageNet上进行了预训练。ImageNet中图像的分辨率都不一样，平均分辨率为$469 \times 387$。[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)是在$224 \times 224$的尺寸上进行fine-tune，然后再将输入维度扩展至$448 \times 448$。对于YOLOv2，我们先直接在$448 \times 448$的尺寸上fine-tune 10个epoch，然后再在目标数据集上继续fine-tune。这一操作使得mAP上升了4%左右。

**Convolutional With Anchor Boxes**

[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)使用全连接层直接预测bounding box的坐标。而[Faster R-CNN](http://shichaoxin.com/2022/04/03/论文阅读-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)的RPN仅使用卷积层来预测bounding box的offset和confidence。相比[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)直接预测bounding box的坐标，预测bounding box的offset会简化问题并使得网络更容易学习。

我们移除了[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)最后的FC层，并使用anchor box机制来预测bounding box。同时，我们也移除了最后一个pooling层，使得网络输出的分辨率更高。此外，object（尤其是大的object）通常位于图像的中心，所以我们希望feature map恰好有一个中心点可以预测这些object。因此，我们将网络的输入从$448\times 448$改为$416\times 416$，这样我们可以得到一个奇数边长的feature map（$13\times 13$）。

此外，对于每一个anchor box，我们都配备了一个条件类别概率：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/3.png)

confidence的定义和算法与[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)一样，即该anchor box存在object的概率。

>个人理解：这样的话，相比[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)，YOLOv2的一个grid cell可以有多个object。

[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)只预测了98个box，但是YOLOv2可以预测上千个box。如果不使用这种anchor box机制，mAP=69.5%，recall=81%；使用anchor box机制后，mAP=69.2%，recall=88%。虽然mAP稍有降低，但是recall提升了很多。

**Dimension Clusters**

我们在使用anchor box时遇到了两个问题。第一个问题是事先会人为设定一个anchor box的维度，然后由网络学习其offset。如果我们事先可以设定一个不错的维度，那么网络学起来会更容易，预测结果也会更好。

因此我们放弃了人为设定初始anchor box的维度，而是通过[k-means聚类](http://shichaoxin.com/2022/03/21/机器学习基础-第三十五课-聚类之原型聚类/#2k均值算法)（基于训练集的bounding box）的方法来自动设定维度。如果我们使用欧式距离进行[k-means聚类](http://shichaoxin.com/2022/03/21/机器学习基础-第三十五课-聚类之原型聚类/#2k均值算法)，则大的box会比小的box产生更大的误差。鉴于我们想要的是更好的IoU分数，因此我们使用的距离计算为：

>个人理解：每个grid cell都可以通过[k-means聚类](http://shichaoxin.com/2022/03/21/机器学习基础-第三十五课-聚类之原型聚类/#2k均值算法)得到k个不同维度的anchor box。

$$d(\text{box},\text{centroid})=1-\text{IOU}(\text{box},\text{centroid})$$

>个人理解：两个box之间的距离计算为1减去这两个box的IoU。两个box如果重叠度越高，那么IoU就会越高，距离就会更小。centroid即为簇的质心，相当于是[k-means聚类](http://shichaoxin.com/2022/03/21/机器学习基础-第三十五课-聚类之原型聚类/#2k均值算法)中的均值向量，只不过这里是一个box。最开始先随机选择k个bounding box作为质心，然后迭代。

我们测试了多个不同的k值，并计算了平均IoU，见Fig2。

>个人理解：可以先计算每个簇内两两box的IoU的均值，然后再计算k个簇的IoU的均值。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/4.png)

根据Fig2，在权衡了recall和模型复杂度后，我们最终选择了k=5。从Fig2右可以看出，聚类的结果（即the cluster centroids）和以前人为设定的anchor box维度很不一样。

我们将[k-means聚类](http://shichaoxin.com/2022/03/21/机器学习基础-第三十五课-聚类之原型聚类/#2k均值算法)得到的平均IoU和其他进行了比较，见表1。表1中第三行（“Anchor Boxes”）为[Faster R-CNN](http://shichaoxin.com/2022/04/03/论文阅读-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)的结果，其一共人为设定了9种不同维度的anchor box，平均IoU为60.9。当我们使用k=5时，平均IoU就达到了61.0，如果使用k=9，平均IoU更是达到了67.2。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/5.png)

**Direct location prediction**

现在我们来说使用anchor box机制遇到的第二个问题：模型不稳定（model instability），尤其是在早期迭代时。这种不稳定来自我们直接预测box的坐标$(x,y)$。在[RPN](http://shichaoxin.com/2022/04/03/论文阅读-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/#31region-proposal-networks)中，其预测的是offset $t_x,t_y$，计算如下：

$$x=(t_x * w_a)-x_a$$

$$y=(t_y * h_a) - y_a$$

例如，如果$t_x=1$，则box的中心向右移动$w_a$；如果$t_x=-1$，则box的中心向左移动$w_a$。

但是，这些公式并没有限制box移动的最终位置，box可能移动到图像的任意一个位置。在随机初始化情况下，模型需要很长时间才能稳定预测出合理的offset。

因此，我们即要预测offset又要保证box不会移动的过远。在YOLOv2中，每个cell预测5个bounding box。每个bounding box包含5个预测值：$t_x,t_y,t_w,t_h,t_o$。我们通过逻辑回归将$t_x,t_y,t_w,t_h$的值限制在0~1之间（和[YOLOv1]()一样），如Fig3所示，假设某cell左上角在全图里的坐标为$(c_x,c_y)$，$\sigma(t_x)$可以简单的理解为$t_x \times cell.width$，同理，$\sigma(t_y)$可理解为$t_y \times cell.height$。$p_w,p_h$为先验box（其实就是聚类得到的anchor box）的宽和高，$t_x,t_y,t_w,t_h$为相对于anchor box的offset，$b_x,b_y,b_w,b_h$为预测得到的box（即经过offset的anchor box）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/6.png)

这样就相当于是将预测的box和对应的anchor box限制在同一个cell里，这使得网络学习更为容易，模型也更加稳定。上一部分的Dimension Clusters和这一部分的Direct location prediction的搭配使用使得mAP提升了将近5%（69.6%到74.4%）。

**Fine-Grained Features**

YOLOv2得到的feature map为$13 \times 13 \times 1024$。对于large object来说，这个维度足够了，但是对于small obejct来说，作者想要更加细粒度的特征。因此，我们把目光瞄准到了上一层的feature map，其维度为$26 \times 26 \times 512$。

我们添加了一个passthrough layer，将$26 \times 26$的卷积层和$13\times 13$的卷积层concat起来。操作流程就是，将$26 \times 26 \times 512$的feature map拆分为4个$13 \times 13 \times 512$的feature map，拆分方法见如下示意图：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/7.png)

然后这4个$13 \times 13 \times 512$的feature map就可以concat为$13 \times 13 \times 2048$的feature map了。这一操作带来了1%的性能提升（74.4%到75.4%）。

>在其他博客上看到的：YOLOv2采用了Darknet-19，对于$26 \times 26 \times 512$的feature map，先通过$1\times 1\times 64$降维到$26 \times 26 \times 64$，然后通过passthrough layer变为$13\times 13 \times 256$，然后和后续卷积层的$13 \times 13 \times 1024$的feature map连接，最后得到$13 \times 13 \times 1280$的feature map。

**Multi-Scale Training**

因为YOLOv2都是卷积层，所以输入图像的大小可以是任意的（个人理解：图像大小不同，最后一个卷积层的feature map维度也不同，所以对应的grid cell的数量也不同，但每一层的卷积核大小都是一样的，因此可以用多尺度的图像去训练这些卷积核的参数）。因此我们尝试用不同尺寸的图像来训练模型以增强鲁棒性。

每10个epoch，我们就重新选择一次图像尺寸。因为我们模型的下采样因子为32，所以我们选择的图像尺寸都是32的倍数：$\\{ 320,352,…,608 \\}$。

这也意味着我们的模型可以预测不同尺寸的图像。尺寸越小，预测速度越快。在PASCAL VOC 2007上的测试结果见表3：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/8.png)

当预测图像尺寸为$288 \times 288$时，YOLOv2可以达到90fps，并且和[Fast R-CNN](http://shichaoxin.com/2022/03/07/论文阅读-Fast-R-CNN/)的mAP差不多。在比较高的分辨率（$544 \times 544$）上，YOLOv2达到了SOTA的mAP（78.6%）。可视化结果见Fig4：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/9.png)

**Further Experiments**

在VOC 2012上训练YOLOv2，测试结果及和其他方法的比较见表4，YOLOv2的mAP和SOTA方法不相上下，但是速度却快得多。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/10.png)

同样，我们也在COCO数据集上进行了训练和测试，结果见表5：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/11.png)

# 3.Faster

我们不但希望我们的检测模型准，还希望它快。在目标检测的很多应用领域中，比如机器人和自动驾驶，都要求预测的低延迟。而YOLOv2就可以做到。

很多检测框架的基础结构都是[VGG-16](http://shichaoxin.com/2021/02/24/论文阅读-VERY-DEEP-CONVOLUTIONAL-NETWORKS-FOR-LARGE-SCALE-IMAGE-RECOGNITION/)。[VGG-16](http://shichaoxin.com/2021/02/24/论文阅读-VERY-DEEP-CONVOLUTIONAL-NETWORKS-FOR-LARGE-SCALE-IMAGE-RECOGNITION/)的优点是分类精度高，缺点是过于复杂。[VGG-16](http://shichaoxin.com/2021/02/24/论文阅读-VERY-DEEP-CONVOLUTIONAL-NETWORKS-FOR-LARGE-SCALE-IMAGE-RECOGNITION/)需要306亿9000万次浮点数运算，且输入必须得是$224 \times 224$。

[YOLOv1](http://shichaoxin.com/2022/05/11/论文阅读-You-Only-Look-Once-Unified,-Real-Time-Object-Detection/)基于[GoogLeNet](http://shichaoxin.com/2021/06/01/论文阅读-Going-deeper-with-convolutions/)设计了自己的网络结构，比[VGG-16](http://shichaoxin.com/2021/02/24/论文阅读-VERY-DEEP-CONVOLUTIONAL-NETWORKS-FOR-LARGE-SCALE-IMAGE-RECOGNITION/)快了很多，一次前向传播需要执行85亿2000万次运算。但是弊端就是精度比使用[VGG-16](http://shichaoxin.com/2021/02/24/论文阅读-VERY-DEEP-CONVOLUTIONAL-NETWORKS-FOR-LARGE-SCALE-IMAGE-RECOGNITION/)差了一些。对于$224\times 224$，single-crop，考虑top-5准确率，在ImageNet上，YOLO自定义的模型得到88%的精度，[VGG-16](http://shichaoxin.com/2021/02/24/论文阅读-VERY-DEEP-CONVOLUTIONAL-NETWORKS-FOR-LARGE-SCALE-IMAGE-RECOGNITION/)可达到90.0%的精度。

**Darknet-19**

我们提出一个新的分类模型作为YOLOv2的基础。和[VGG](http://shichaoxin.com/2021/02/24/论文阅读-VERY-DEEP-CONVOLUTIONAL-NETWORKS-FOR-LARGE-SCALE-IMAGE-RECOGNITION/)类似，我们主要使用$3\times 3$的filter，并且在每个pooling层后将channel数加倍。此外，我们的模型也使用了$1 \times 1$的卷积来降维。还使用了[BatchNorm](http://shichaoxin.com/2021/11/02/论文阅读-Batch-Normalization-Accelerating-Deep-Network-Training-by-Reducing-Internal-Covariate-Shift/)。

我们将这个模型称为Darknet-19，共有19个卷积层和5个maxpooling层。具体结构见表6。Darknet-19处理一张图像只需要55亿8000万次运算，在ImageNet上，top-1精度为72.9%，top-5精度为91.2%。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/12.png)

**Training for classification**

我们在标准的ImageNet1000类别的分类数据集上训练Darknet-19，一共训练了160个epoch，使用[SGD](http://shichaoxin.com/2020/02/20/深度学习基础-第十五课-mini-batch梯度下降法/)，初始学习率为0.1，polynomial rate decay with a power of 4（个人理解：一个用于[学习率衰减](http://shichaoxin.com/2020/03/23/深度学习基础-第二十课-学习率衰减/)的参数），[weight decay](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#314l1正则化和l2正则化的区别)=0.0005，[momentum](http://shichaoxin.com/2020/03/05/深度学习基础-第十七课-Momentum梯度下降法/)=0.9。训练过程中，我们使用了常规的数据扩展方法，包括：random crops、rotations、hue、saturation、exposure shifts。

在使用$224 \times 224$大小的图像对模型进行预训练后，我们使用了$448 \times 448$大小的图像进行了fine-tune。fine-tune的参数和上述一样，但是只训练了10个epoch，学习率为$10^{-3}$。此举使得我们的模型top-1精度为76.5%，top-5精度为93.3%。

**Training for detection**

我们将这个模型稍作修改用于目标检测，我们移除了最后一个卷积层，取而代之的是在最后新添加三个$3\times 3$的卷积层（filter数量为1024），最后接一个$1 \times 1$卷积层用于将维度降到我们需要的大小。比如对于VOC数据集，每个grid cell预测5个box，每个box对应5个坐标值以及20个类别概率，因此，维度应该是125（$5 \times (5+20)$）。此外，我们也添加了passthrough机制。

我们一共训练了160个epoch，初始学习率为$10^{-3}$，并且在第60个和第90个epoch的时候，学习率缩小10倍。[weight decay](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#314l1正则化和l2正则化的区别)=0.0005，[momentum](http://shichaoxin.com/2020/03/05/深度学习基础-第十七课-Momentum梯度下降法/)=0.9。我们使用了和上文类似的数据扩展方法和训练策略。

# 4.Stronger

我们提出了一种联合训练机制，使得模型可以同时在目标检测和图像分类数据集上进行训练。

在训练过程中，两种数据集被混在一起。当图像是来自目标检测数据集时，我们就反向传播YOLOv2整个loss function；而当图像是来自图像分类数据集时，我们只反向传播loss function中分类部分的loss。

这个方法也存在一些挑战。目标检测数据集中只有常见的object及其比较宽泛的label，比如dog和boat等。但是对于图像分类数据集，ImageNet会将dog分为一百多个不同的品种。因此，我们就需要merge这些不同的label。

如果我们只是简单的把dog作为一个类别，金毛作为另一个类别，这显然是不合理的，因为金毛是狗的一个品种，这两个类别并不是完全互斥的。

**Hierarchical classification**

ImageNet的标签来自WordNet。在WordNet中，Norfolk terrier的层级关系为：canine->dog->hunting dog->terrier->Norfolk terrier。这种结构正是我们所需要的。

但是WordNet的结构是一个有向图，并不是一个树形结构。这是因为语言系统的复杂性，比如dog既可以属于canine，也可以属于domestic animal。因此我们并没有使用全部的WordNet图结构，而是根据ImageNet所用到的标签，将其简化为一个hierarchical tree，我们称之为WordTree。

具体做法是首先将只有一条路径的名词挑选出来，而对于有多条路径的名词，本着尽可能少的增加树的路径的原则进行选择。

使用WordTree时，我们会计算每个下位词的条件概率，比如：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/13.png)

如果我们想求某个节点的绝对概率，可以将其在WordTree里的路径上的条件概率相乘，比如：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/14.png)

对于分类目的，我们假定图像一定包含object，即：

$$Pr(\text{physical object})=1$$

为了验证这个方法，我们使用1000个类别的ImageNet，基于WordTree训练了一个Darknet-19模型。为了构建WordTree1k，我们将类别从1000扩充到了1369，见Fig5。在训练过程中，以Norfolk terrier为例，根据WordTree1k，其会有多个GT标签，比如Norfolk terrier、terrier、dog、mammal等。不像ImageNet，通常都使用一个softmax函数，对于WordTree1k，我们使用多个softmax函数，见Fig5（个人理解：级联softmax。比如，先用一个softmax预测该图像属于和Norfolk terrier一层的所有类别的概率，再用另一个softmax预测该图像属于和terrier一层的所有类别的概率，然后再用一个新的softmax预测图像属于和hunting dog一层的所有类别的概率，剩下的以此类推）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/15.png)

使用和之前一样的训练参数，Darknet-19的top-1准确率为71.9%，top-5准确率为90.4%。尽管增加了额外的369个概念，但模型的准确率只是略有下降，并且这种方法还有一些其他的优点。比如，模型有时候可能不确定dog的品种，但其至少能以很高的置信度预测其为dog，只是dog下位词的置信度较低。

在检测时，我们通过YOLOv2得到存在object的概率$Pr(\text{physical object})$。然后根据WordTree和级联softmax，计算出属于各个类别的概率。

>个人理解：将Fig5下所示的结构作为每个bounding box的类别概率部分，大概是这个意思：
>
>![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/16.png)

**Dataset combination with WordTree**

我们可以通过WordTree将多个数据集合理的组合在一起。例如Fig6，我们将ImageNet和COCO数据集的标签通过WordTree合并在了一起：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/YOLOv2/17.png)

**Joint classification and detection**

现在我们可以通过WordTree，在分类和检测的联合数据集上进行模型的训练。我们合并了COCO检测数据集、ImageNet9000个类别的分类数据集、ImageNet检测数据集，最终得到9418个类别。因为ImageNet的数据量远大于COCO数据集，出于数据平衡的目的，我们对COCO数据集进行了oversampling，大致使ImageNet和COCO的数据量比值在4:1左右。

基于上述数据集，使用YOLOv2框架，我们训练了YOLO9000。注意，每个grid cell只用了3个anchor box以减小输出的size。对于检测数据，我们就正常反向传播loss。对于分类数据，我们就只反向传播该层以及上位层的类别loss（这里的层指的是WordTree中的层）。

对于分类数据，虽然只计算类别loss，但是还需要注意几点。我们先找到哪个bounding box中将GT类别的概率预测的最高，我们就只计算这一个bounding box的类别loss（别的bounding box不管）。我们还假设预测的box和GT的IoU至少为0.3，只有满足这个假设，我们才反向传播objectness loss（个人理解：如果预测的box和GT的IoU过低，此时计算类别loss意义不大，因为box内很可能就没有object，只有预测的box和GT的IoU达到一定阈值，此时再判断box里包含的object属于哪个类别才有意义）。

>个人理解：对于分类数据，只针对GT类别概率最高且和GT box的IoU大于等于0.3的bounding box做类别loss的反向传播。

通过这种联合训练，YOLO9000通过COCO检测数据集学到了定位object的能力，通过ImageNet分类数据集学到了区分多种object所属类别的能力。

我们在ImageNet检测任务上评估了YOLO9000。ImageNet检测任务和COCO只有44个共有类别，这意味着对于测试数据集（即ImageNet检测数据集），模型在训练时只在分类数据中见过这些标签，在检测数据集中大部分标签都没见过。最终，YOLO9000得到了19.7的mAP。如果只考虑训练检测数据集中没有的156个类别，那么YOLO9000依然可以达到16.0的mAP。这一成绩比DPM要好。此外，YOLO9000支持实时预测9000个类别。

在后续结果分析中我们还发现，对于和COCO检测数据集中相似的类别，即使COCO检测数据集中不包含这些类别，YOLO9000也可以很好的预测（比如动物的品种等）。但是对于COCO检测数据集中没有任何相似类别的类别（比如衣服，COCO检测数据集中不包含这种类别，也没有相似类别），YOLO9000就预测的不是很好。

# 5.Conclusion

我们介绍了YOLOv2和YOLO9000，都是real-time detection system。YOLOv2达到了SOTA的水平，并且更快，在速度和精度上有很好的tradeoff。

YOLO9000是一个real-time的框架，可以检测超过9000个类别。我们通过WordTree在ImageNet和COCO上进行了联合训练。YOLO9000缩小了检测数据集和分类数据集之间的gap。

# 6.原文链接

👽[YOLO9000：Better, Faster, Stronger](https://github.com/x-jeff/AI_Papers/blob/master/YOLO9000：%20Better%2C%20Faster%2C%20Stronger.pdf)

# 7.参考资料

1. [YOLO v2详细解读](https://blog.csdn.net/weixin_43694096/article/details/123523679)