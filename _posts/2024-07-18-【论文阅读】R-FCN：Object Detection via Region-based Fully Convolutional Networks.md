---
layout:     post
title:      【论文阅读】R-FCN：Object Detection via Region-based Fully Convolutional Networks
subtitle:   R-FCN，OHEM
date:       2024-07-18
author:     x-jeff
header-img: blogimg/20221002.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

>源码地址：[R-FCN](https://github.com/daijifeng001/r-fcn)。

[SPP-net](https://shichaoxin.com/2022/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Spatial-Pyramid-Pooling-in-Deep-Convolutional-Networks-for-Visual-Recognition/)、[Fast R-CNN](https://shichaoxin.com/2022/03/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fast-R-CNN/)、[Faster R-CNN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)等一系列目标检测框架可通过RoI pooling层划分为两个子网络：1）RoI pooling层之前，是和RoI无关的、共享的全卷积子网络；2）RoI pooling层之后，是基于RoI的子网络，之间互不共享。以[SPP-net](https://shichaoxin.com/2022/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Spatial-Pyramid-Pooling-in-Deep-Convolutional-Networks-for-Visual-Recognition/)为例，第一个子网络是一个以spatial pooling layer为结束的卷积网络，第二个子网络是多个fc层。因此，我们就可以用分类网络中的最后一个spatial pooling layer作为目标检测网络中的RoI pooling层。

但是最近一些SOTA的图像分类网络，比如[ResNet](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)和GoogLeNet（[Inception-v1](https://shichaoxin.com/2021/06/01/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Going-deeper-with-convolutions/)、[Inception-v2/v3](https://shichaoxin.com/2021/11/29/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Rethinking-the-Inception-Architecture-for-Computer-Vision/)），都被设计成全卷积（只有最后一层是全连接，在迁移到目标检测任务上时会被移除）。很自然的就会想到，在目标检测框架中使用所有的卷积层作为共享的卷积子网络，而基于RoI的子网络则没有隐藏层。但这种方案的检测精度非常低，和很高的分类精度不符。为了解决这个问题，在[ResNet论文](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)中，[Faster R-CNN检测器](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)的RoI pooling层被不自然的插入到两组卷积层之间——这创建了一个更深的RoI子网络，提高了精度，但由于每个RoI的计算是非共享的，因此速度较慢。

我们认为上述不自然的设计是由于增加图像分类的平移不变性与满足目标检测的平移变化性之间的矛盾造成的。一方面，图像分类任务更倾向于平移不变性——即不会关心和识别目标在图像中的移动，因此，具有平移不变性的深层全卷积网络是首选。另一方面，目标检测任务则需要对移动的目标进行定位。例如，我们需要对移动的目标做出有意义的响应，即生成相应的bbox。

在本文中，针对目标检测任务，我们提出了R-FCN（Region-based Fully Convolutional Network）。我们的网络由共享的、全卷积框架组成，就像[FCN](https://shichaoxin.com/2022/01/31/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fully-Convolutional-Networks-for-Semantic-Segmentation/)一样。核心思路见Fig1：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/1.png)

在Fig1中，输入图像首先进入一个全卷积网络，然后通过一个专门的卷积得到position-sensitive score map。接着基于position-sensitive score map进行RoI pooling。

基于region的检测器之间的比较见表1：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/2.png)

# 2.Our approach

👉**Overview.**

遵循[R-CNN](https://shichaoxin.com/2021/09/20/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Rich-feature-hierarchies-for-accurate-object-detection-and-semantic-segmentation/)，我们采用了two-stage的目标检测策略：1）region proposal；2）region classification。我们使用[RPN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/#31region-proposal-networks)生成候选region。[RPN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/#31region-proposal-networks)和R-FCN之间共享特征。整体框架见Fig2。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/3.png)

给定proposal region（即RoI），R-FCN会预测这个RoI的类别（目标类别或背景）。在R-FCN中，所有可学习的权重层都是卷积的，并且是在整个图像上计算的。最后一个卷积层针对每个类别输出$k^2$个position-sensitive score map，算上背景，一共是$C+1$个类别，所以最后一个卷积层的输出通道数为$k^2(C+1)$。

👉**Backbone architecture.**

R-FCN的backbone使用[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)。我们只使用[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)的卷积层（共100个卷积层）来计算feature map，去除了[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)中的全局平均池化层和1000类别的fc层。[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)在ImageNet上进行了预训练。[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)最后一个卷积block的输出维度是2048-d，我们额外添加了一个1024-d的$1\times 1$卷积用于降低维度。在此之后，我们使用了一个通道数为$k^2(C+1)$的卷积层来生成score map。

👉**Position-sensitive score maps & Position-sensitive RoI pooling.**

我们将每个RoI划分为$k \times k$个bin。如果RoI的大小为$w \times h$，那么每个bin的大小约为$\frac{w}{k} \times \frac{h}{k}$。在我们的方法中，最后一个卷积层为每个类别生成了$k^2$个score map（个人注解：每个类别的每个bin都对应一个score map）。那么对于第$(i,j)$个bin（$0 \leqslant i,j \leqslant k-1$），基于score map的RoI pooling可表示为：

$$r_c(i,j \mid \Theta) = \sum_{(x,y)\in \text{bin}(i,j)} z_{i,j,c} (x+x_0, y+y_0 \mid \Theta) / n \tag{1}$$

$r_c(i,j)$是在第$c$个类别上，第$(i,j)$个bin经过池化后得到的值。$z_{i,j,c}$是$k^2(C+1)$个score map中的一个。$(x_0,y_0)$是RoI左上角的坐标。$n$是这个bin里的像素点数量。$\Theta$表示这个网络中所有可学习的参数。第$(i,j)$个bin的范围：

$$\lfloor i \frac{w}{k} \rfloor \leqslant x < \lceil (i+1)\frac{w}{k} \rceil$$

$$\lfloor j \frac{h}{k} \rfloor \leqslant y < \lceil (j+1) \frac{h}{k} \rceil$$

如Fig1所示，每个颜色代表一个bin。式(1)执行的是平均池化，如果想执行max pooling也是可以的。

对每个类别（即通道），我们得到了$k^2$个position-sensitive score，我们将这些分数求平均，对每个RoI来说，最终得到了一个$(C+1)$维的向量：

$$r_c(\Theta) = \sum_{i,j} r_c (i,j \mid \Theta)$$

>个人注解：求平均应该再除个$k^2$。

然后计算softmax：

$$s_c(\Theta) = e^{r_c(\Theta)} / \sum_{c'=0}^C e^{r_{c'}(\Theta)}$$

这样就能得到该RoI属于各个类别的概率了。在训练时，此处使用交叉熵损失。

预测bbox的方式和上述一样，在backbone后再添加一个并行的分支（个人注解：此时，相当于一共有3个并行的分支：RPN、类别预测分支、bbox预测分支）：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/4.png)

position-sensitive score map的个数变为$4k^2$。和预测类别分支的操作一样，最终得到一个4维向量，用于表示bbox（参数含义同[Fast R-CNN](https://shichaoxin.com/2022/03/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fast-R-CNN/)）：

$$t= (t_x,t_y,t_w,t_h)$$

需要注意的是，这里的bbox预测分支是不考虑类别的，即类别无关的。但如果想针对每个类别都预测一个bbox也是可以的，这样的话，position-sensitive score map的数量就是$4k^2 C$。

在RoI层之后就没有要学习的层了，这使得基于region的计算几乎是无成本的，这加快了训练和推理的速度。

👉**Training.**

有了预先计算的region proposal，可以很容易的对R-FCN进行端到端的训练。损失函数的定义和[Fast R-CNN](https://shichaoxin.com/2022/03/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fast-R-CNN/#23fine-tuning-for-detection)一样。和GT box的IoU大于0.5的RoI被视为正样本，否则为负样本。

我们在训练过程中可以很容易的应用OHEM（online hard example mining）。我们对每个RoI的计算量可以忽略不计，因此example mining几乎是无成本。假设每张图片有$N$个proposal，在前向传播中，我们会计算每个proposal的loss。然后，我们按照loss大小对所有的RoI进行排序，选择loss最大的$B$个RoI。反向传播基于这$B$个RoI进行。由于我们对每个RoI的计算量可以忽略不计，因此前向传播的时间几乎不受$N$的影响，与此相反，OHEM Fast R-CNN的训练时间可能会增加一倍。

>个人注解：
>
>简单介绍下OHEM，其出自论文“A. Shrivastava, A. Gupta, and R. Girshick. Training region-based object detectors with online hard example mining. In CVPR, 2016.”。OHEM是一种用于提升机器学习模型训练效果的技术，尤其在计算机视觉和目标检测任务中常用。其核心思想是动态选择训练过程中最难的样本进行训练，而不是使用所有样本。这样做的目的是提高模型对难以识别的样本的泛化能力和鲁棒性。
>
>背景：
>
>在训练深度学习模型时，数据集中通常包含大量容易分类的样本以及少量难以分类的样本。使用所有样本进行训练可能会导致模型更关注容易分类的样本，从而忽视了那些难以分类的样本。这会导致模型在训练集上的性能良好，但在测试集或实际应用中表现不佳，特别是在处理难以分类的样本时。
>
>核心思想：
>
>OHEM的基本思想是通过在线方式（即在训练过程中）选择那些对当前模型来说较难分类的样本，优先对这些样本进行训练，从而提高模型的性能。
>
>实现步骤：
>
>1. 前向传播：在每个训练批次中，首先通过当前模型对所有样本进行前向传播，计算出预测结果和损失值。
>2. 样本选择：根据损失值对样本进行排序，选择损失值较大的部分样本作为“困难样本”。通常，会选择损失值排名前$k$的样本，$k$是一个预先定义的超参数，决定了每个批次中选择多少比例的困难样本。
>3. 后向传播：对选定的困难样本进行后向传播和参数更新。
>4. 重复：重复上述过程，直到完成所有训练迭代。

weight decay为0.0005，momentum为0.9。使用单尺度训练：将图片的短边resize到600个像素。每块GPU处理一张图像，设置$B=128$。训练使用了8块GPU。fine-tune R-FCN使用的学习率为0.001（20k个mini-batch）和0.0001（10k个mini-batch），基于VOC数据集。对于RPN，我们采用了[Faster R-CNN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/#32sharing-features-for-rpn-and-fast-r-cnn)中的4步训练法。

👉**Inference.**

如Fig2所示，首先计算RPN和R-FCN共享部分的feature map。然后RPN生成RoI，接着R-FCN基于RoI计算类别概率和bbox。为了和[Faster R-CNN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)公平比较，在推理阶段，我们评估了300个RoI。后处理使用了NMS（IoU阈值为0.3）。

👉**À trous and stride.**

得益于[FCN](https://shichaoxin.com/2022/01/31/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fully-Convolutional-Networks-for-Semantic-Segmentation/)，我们的方法修改后也可用于语义分割。我们将[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)的32倍下采样降低到了16倍下采样，以增加score map的分辨率。通过把conv5_1的步长改为1实现16倍下采样，并在conv5阶段使用空洞卷积来弥补降低的步长。为了公平比较，RPN的计算基于conv1~conv4（即这部分和R-FCN共享），这样RPN就不受空洞卷积的影响了。下表是一个相关的消融实验，其中$k\times k = 7 \times 7$，没有使用OHEM：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/5.png)

>à trous表示使用了空洞卷积。

👉**Visualization.**

在Fig3和Fig4中，我们可视化了R-FCN学到的position-sensitive score map（$k \times k = 3 \times 3$）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/6.png)

# 3.Related Work

不再赘述。

# 4.Experiments

## 4.1.Experiments on PASCAL VOC

我们在有20个目标类别的PASCAL VOC上进行了实验。我们在VOC 2007 trainval和VOC 2012 trainval（即“07+12”）上进行了训练，在VOC 2007 test上进行了评估。目标检测精度使用mAP（mean Average Precision）进行评估。

👉**Comparisons with Other Fully Convolutional Strategies**

我们使用[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)作为backbone测试了以下全卷积策略（或“几乎”全卷积策略，每个RoI只有一个分类器fc层）：

* **Naïve Faster R-CNN.**：[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)所有的卷积层都被用来计算共享的feature map，把RoI pooling放在最后一个卷积层之后（即conv5之后），对每个RoI应用一个21类别的fc层。为了公平比较，使用了空洞卷积。
* **Class-specific RPN.**：RPN的训练遵循[Faster R-CNN论文](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)，唯一的区别在于将分类层的2类（前景或背景）改为了21类。为了公平比较，class-specific RPN在[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)的conv5使用了空洞卷积。
* **R-FCN without position-sensitivity.**：通过设置$k=1$可将R-FCN中的position-sensitivity移除。这相当于对每个RoI进行global pooling。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/7.png)

[ResNet原文](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)中用ResNet-101实现的standard Faster R-CNN达到了76.4%的mAP（见表3），其RoI pooling层在conv4和conv5之间。而naïve Faster R-CNN（将RoI pooling放在conv5之后）的mAP掉到了68.9%（见表2）。这个比较结果说明了，对于Faster R-CNN，在层之间插入RoI pooling以强调空间信息是非常重要的。

R-FCN without position-sensitivity因模型无法收敛导致fail。

👉**Comparisons with Faster R-CNN Using ResNet-101**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/8.png)

对于表3，我们都使用$k \times k = 7 \times 7$。更多的比较见表4。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/9.png)

对于表4，在multi-scale训练中，在每个训练迭代中，我们随机将图像的短边resize到$\\{ 400,500,600,700,800 \\}$个像素。在single-scale训练中，图像的短边固定为600个像素。可视化结果见下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/14.png)

在PASCAL VOC 2012上的比较见表5。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/10.png)

更细节的检测结果见表7和表8。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/16.png)

👉**On the Impact of Depth**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/11.png)

👉**On the Impact of Region Proposals**

都使用[ResNet-101](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)作为backbone，测试使用不同的proposal生成方法：[Selective Search（SS）](https://shichaoxin.com/2021/10/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Selective-Search-for-Object-Recognition/)、Edge Boxes（EB）、[RPN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/12.png)

## 4.2.Experiments on MS COCO

我们还在有80个类别的MS COCO数据集上进行了评估。我们的实验包括80k张图像的train set、40k张图像的val set和20k张图像的test-dev set。前90k次迭代的学习率为0.001，后30k次迭代的学习率为0.0001，mini-batch size=8。我们还把[Faster R-CNN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)中的4步训练法扩展到了5步（即在最后多加了一步用于训练RPN），这样做略微提高了在该数据集上的精度。我们还发现，只使用前两步训练也可以获得相对较好的精度，但没有特征共享。

考虑到COCO数据集的目标尺度跨度更广，测试阶段的multi-scale使用了$\\{ 200,400,600,800,1000 \\}$（见表6最后一行）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/13.png)

可视化结果见下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/RFCN/15.png)

# 5.Conclusion and Future Work

R-FCN是一个简单且高效的目标检测方法。

# 6.原文链接

👽[R-FCN：Object Detection via Region-based Fully Convolutional Networks](https://github.com/x-jeff/AI_Papers/blob/master/2024/R-FCN：Object%20Detection%20via%20Region-based%20Fully%20Convolutional%20Networks.pdf)