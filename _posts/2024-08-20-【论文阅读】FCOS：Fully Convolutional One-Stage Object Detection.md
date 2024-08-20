---
layout:     post
title:      【论文阅读】FCOS：Fully Convolutional One-Stage Object Detection
subtitle:   FCOS
date:       2024-08-20
author:     x-jeff
header-img: blogimg/20191226.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

>代码：[FCOS](https://github.com/tianzhi0549/FCOS/)。

anchor-based检测器有以下一些缺点：

1. 检测性能对anchor box的大小、长宽比以及数量很敏感。
2. 即使anchor box经过了精心设计，但由于anchor box的大小和长宽比是保持不变的，所以在处理形状变化较大的目标时也会遇到困难，特别是对于小目标。此外，预设的anchor box也阻碍了检测器的泛化能力。
3. 为了获得更高的recall rate，anchor box需要被密集的放置在输入图像上（比如在[FPN](https://shichaoxin.com/2023/12/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Feature-Pyramid-Networks-for-Object-Detection/)中，将输入图像的短边缩放到800，对于一张输入图像，我们需要超过180K个anchor box）。在训练阶段，大多数anchor box都被标记为负样本。过多的负样本加剧了训练中正负样本的不平衡。
4. anchor box还涉及复杂的计算，比如IoU的计算。

在本文中，我们首次证明了，更简单的基于[FCN](https://shichaoxin.com/2022/01/31/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fully-Convolutional-Networks-for-Semantic-Segmentation/)的检测器比基于anchor的检测器具有更好的性能。

>个人注解：FCOS是anchor-free的。

# 2.Related Work

不再赘述。

# 3.Our Approach

## 3.1.Fully Convolutional One-Stage Object Detector

将CNN backbone第$i$层的feature map记为$F_i \in \mathbb{R}^{H \times W \times C}$，$s$为到这一层累积的总步长。将一张输入图像的GT box记为$\\{B_i \\}$，其中，$B_i = (x_0^{(i)},y_0^{(i)},x_1^{(i)},y_1^{(i)},c^{(i)}) \in \mathbb{R}^4 \times \\{ 1,2,...,C \\}$。$(x_0^{(i)},y_0^{(i)})$和$(x_1^{(i)},y_1^{(i)})$分别是bbox左上角点和右下角点的坐标。$c^{(i)}$是bbox内目标的所属类别。$C$是总的类别数目，对于MS-COCO数据集来说，$C=80$。

对于$F_i$中的任意一点$(x,y)$，其在输入图像中对应的感受野的中心坐标近似为$(\lfloor \frac{s}{2} \rfloor+xs,\lfloor \frac{s}{2} \rfloor+ys)$。基于anchor的方法将输入图像上的位置（location，个人注解：即像素点）视为（多个）anchor box的中心，并且参考这些anchor box来回归目标bbox，而我们直接在该位置上回归目标bbox。换句话说，我们的检测器直接将位置视为训练样本，而不是像基于anchor方法那样将anchor box视为训练样本，这与语义分割中的[FCN](https://shichaoxin.com/2022/01/31/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fully-Convolutional-Networks-for-Semantic-Segmentation/)相同。

具体来说，如果位置$(x,y)$落在任意一个GT box内，它就是正样本，其类别标签$c^\*$就是GT box的类别。否则就是负样本，则$c^\*=0$（背景类别）。除了类别标签，还有一个四维向量$\mathbf{t}^\*=(l^\*,t^\*,r^\*,b^\*)$作为bbox的回归目标。其中，$l^\*,t^\*,r^\*,b^\*$为该位置到bbox四个边界的距离，如Fig1左所示。如果一个位置落入多个bbox内，则视为模糊样本（ambiguous sample）。对于这种情况，我们简单的选择面积最小的bbox作为回归目标。在下一个章节，我们将展示通过多层级预测，可以显著减少模糊样本的数量，从而几乎不影响检测性能。如果位置$(x,y)$匹配上了bbox $B_i$，则训练时的回归目标为：

$$l^* = x-x_0^{(i)}, \  t^*=y-y_0^{(i)} \\ r^* = x_1^{(i)}-x, \  b^* = y_1^{(i)}-y \tag{1}$$

值得注意的是，FCOS可以利用尽可能多的前景样本来训练回归器。这与基于anchor的方法不同，后者只将与GT box有足够高IoU的anchor box视为正样本。我们认为，这可能是FCOS优于基于anchor检测器的原因之一。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/1.png)

👉**Network Outputs.**

网络的最后一层输出一个80D的向量$\mathbf{p}$用于预测类别标签，和一个4D的向量$\mathbf{t} = (l,t,r,b)$用于预测bbox。我们没有训练一个多类别分类器，而是训练了$C$个二分类器。和[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)类似，分类分支和回归分支都包含4个卷积层。此外，由于回归目标总是正的，所以我们使用$\exp (x)$将数映射到$(0,\infty)$。相比流行的基于anchor的方法，比如[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)和[Faster R-CNN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)（每个位置使用9个anchor box），FCOS输出的变量少了9倍。

👉**Loss Function.**

$$L(\{\mathbf{p}_{x,y}\},\{\mathbf{t}_{x,y}\})=\frac{1}{N_{pos}}\sum_{x,y}L_{cls}(\mathbf{p}_{x,y},c^*_{x,y})+\frac{\lambda}{N_{pos}}\sum_{x,y}\mathbb{I}_{\{c^*_{x,y}>0\}}L_{reg}(\mathbf{t}_{x,y},\mathbf{t}^*_{x,y})\tag{2}$$

其中，$L_{cls}$使用[focal loss](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)，$L_{reg}$使用[IoU loss](https://shichaoxin.com/2024/08/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-UnitBox-An-Advanced-Object-Detection-Network/)。$N_{pos}$表示阳性样本的数量，$\lambda$是平衡权重，本文设为1。求和会计算$F_i$上的所有位置。如果$c^\*\_i>0$，则$\mathbb{I}\_{\{c^\*_i>0\}}$为1，否则为0。

👉**Inference.**

给定一个输入图像，网络输出$F_i$上每个位置对应的分类分数$\mathbf{p}\_{x,y}$和回归预测$\mathbf{t}\_{x,y}$。我们将$\mathbf{p}\_{x,y}>0.05$的视为阳性样本。

## 3.2.Multi-level Prediction with FPN for FCOS

在这里，我们展示了如何通过使用带有[FPN](https://shichaoxin.com/2023/12/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Feature-Pyramid-Networks-for-Object-Detection/)的多层级预测来解决FCOS可能存在的两个问题：

1. 最终feature map的较大步长（比如$16\times$，个人注解：下采样倍数）会导致较低的BPR（best possible recall，即检测器所能达到的recall rate上限）。对于基于anchor的检测器，由于较大步长导致的低recall rate，可以通过降低判定正样本anchor box所需的IoU阈值来进行一定程度上的补偿。而我们通过实验证明，即使使用较大的步长，基于[FCN](https://shichaoxin.com/2022/01/31/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fully-Convolutional-Networks-for-Semantic-Segmentation/)的FCOS仍然可以获得好的BPR，甚至优于官方实现Detectron中的基于anchor的检测器[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)的BPR（见表1）。因此，对FCOS来说，BPR不是问题。此外，通过使用多层级[FPN](https://shichaoxin.com/2023/12/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Feature-Pyramid-Networks-for-Object-Detection/)预测，BPR可以进一步提高，达到与[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)所能实现的最佳BPR相当的水平。
2. GT box之间的重叠可能会导致模糊性，即在重叠区域内，某个位置应该属于哪个GT box？这种模糊性会导致基于[FCN](https://shichaoxin.com/2022/01/31/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fully-Convolutional-Networks-for-Semantic-Segmentation/)检测器性能下降。在本文，我们展示了这种模糊性可以通过多层级预测大大缓解，基于[FCN](https://shichaoxin.com/2022/01/31/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Fully-Convolutional-Networks-for-Semantic-Segmentation/)的检测器在性能上可以与基于anchor的检测器相当，甚至更好。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/2.png)

FCOS的框架如Fig2所示。分别在$C3,C4,C5$后接一个$1\times 1$卷积得到$P3,P4,P5$。在$P5$的基础上，使用一个步长为2的卷积层得到$P6$，基于$P6$，再使用一个步长为2的卷积层得到$P7$。$P3,P4,P5,P6,P7$的步长分别为$8,16,32,64,128$。对于所有特征层级中的每一个位置，我们先计算回归目标$l^\*,t^\*,r^\*,b^\*$。如果满足$\max (l^\*,t^\*,r^\*,b^\*)>m_i$或$\max (l^\*,t^\*,r^\*,b^\*)<m_{i-1}$，则该位置视为负样本，不再需要回归bbox。$m_i$是第$i$个特征层级需要回归的最大距离。本文中，$m_2,m_3,m_4,m_5,m_6,m_7$分别设置为$0,64,128,256,512,\infty$。由于不同尺寸的目标被分配到不同的特征层级，而大部分的重叠发生在尺寸差异较大的目标之间。如果一个位置，即使使用了多层级预测，仍然被分配多个GT box，我们会简单的选择面积最小的GT box。

head在不同特征层级之间是共享的，这不仅提升了检测效率，还提高了检测精度。但是，我们发现不同的特征层级需要回归不同的尺寸范围（比如$P3$的尺寸范围是$[0,64]$，$P4$的尺寸范围是$[64,128]$），因此，对不同特征层级直接使用完全一样的head是不合理的。所以，我们将标准的$\exp(x)$替换为了$\exp(s_ix)$，对每个$P_i$都有一个可学习的参数$s_i$用于自动调整，这一改动也略微提升了检测性能。

## 3.3.Center-ness for FCOS

在使用了多层级预测之后，FCOS仍然和基于anchor的检测器有差距。我们观察到这是由于在远离目标中心的位置产生了很多低质量的预测bbox。

我们在不引入任何超参数的情况下，采用了一种高效的策略来抑制这些低质量的检测框。如Fig2所示，我们添加了一个center-ness分支，用于预测位置的“center-ness”。center-ness表示该位置到对应目标中心位置的归一化距离，如Fig7所示。center-ness的计算为：

$$\text{centerness}^*=\sqrt{\frac{\min (l^*,r^*)}{\max (l^*,r^*)} \times \frac{\min (t^*,b^*)}{\max (t^*,b^*)}} \tag{3}$$

>论文提交后，作者在后续实验中发现，在MS-COCO数据集上，如果把center-ness分支移到和回归分支并行的位置上，AP可以进一步提升。
>
>个人注解：式(3)是训练时用来计算loss的，在推理阶段，center-ness的值可直接从center-ness分支获得，本文第4.1.2部分的实验结果也证明了这种方式的性能是最好的。

我们使用$\text{sqrt}$来降低center-ness的衰减速度。center-ness的范围从0到1，使用[BCE loss](https://shichaoxin.com/2024/01/14/YOLO%E7%B3%BB%E5%88%97-YOLOv5/#51compute-losses)训练。其loss被加在式(2)中。在推理阶段，最终的分数（即用来对检测框排序的分数）为预测的center-ness乘上对应的分类分数（个人注解：这个思路和[IoU aware loss](https://shichaoxin.com/2024/08/16/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-IoU-aware-Single-stage-Object-Detector-for-Accurate-Localization/)一样）。从而，center-ness可以降低远离目标中心的bbox的分数。这样，高分类分数但低质量的bbox就会在最终的NMS过程中被过滤掉，从而显著提升了检测性能。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/4.png)

如Fig3的热图所示，越靠近bbox中心，颜色越红，center-ness越接近1；越靠近bbox边缘，颜色越蓝，center-ness越接近0。

除了center-ness，还有另外一种可选的方法，就是只把GT box中心部分所在位置视为正样本。在论文提交后，在后续研究[FCOS\_PLUS](https://github.com/yqyao/FCOS_PLUS)中，我们发现结合这两种方法可以达到更好的性能。实验结果见表3。

# 4.Experiments

实验在COCO数据集上进行。训练集为COCO trainval35k（115K张图像），验证集为minival（5K张图像，用于消融实验）。测试集为test-dev（20K张图像）。

👉**Training Details.**

除非特殊说明，均使用[ResNet-50](https://shichaoxin.com/2022/01/07/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Deep-Residual-Learning-for-Image-Recognition/)作为backbone，使用和[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)一样的超参数。使用SGD训练了90K次迭代，初始学习率为0.01，minibatch size=16。在第60K和第80K次迭代时，学习率除以10。weight decay=0.0001，momentum=0.9。backbone在ImageNet上进行了预训练。新添加层的初始化同[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)。除非特殊说明，将输入图像的短边resize到800个像素，长边小于或等于1333个像素。

👉**Inference Details.**

我们首先将输入图像喂给网络，得到预测的bbox及其对应的预测类别。除非特殊说明，接下来的后处理以及所用的超参数都和[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)保持一致。推理所用的输入图像大小和训练所用的一样。当然，如果我们仔细的调整这些超参数，模型的性能可能会进一步提高。

## 4.1.Ablation Study

### 4.1.1.Multi-level Prediction with FPN

针对第3.2部分提到的两个可能问题的消融实验。

👉**Best Possible Recalls.**

BPR的定义为检测器最多能recall到的GT box数量和所有GT box数量的比值。如果在训练阶段，GT box至少被分配给一个样本（在FCOS中，样本指的是像素点位置；在anchor-based方法中，样本指的是anchor），则认为该box被recall到了。如表1所示，只有特征层级$P4$（步长为16）的话（即不使用[FPN](https://shichaoxin.com/2023/12/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Feature-Pyramid-Networks-for-Object-Detection/)），FCOS的BPR为95.55%。而Detectron官方实现的[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)的BPR只有90.92%，且$\text{IOU} \geqslant 0.4$才认为anchor和GT box匹配成功。如果取消IoU阈值的限制，[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)的BPR达到了99.23%。在[FPN](https://shichaoxin.com/2023/12/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Feature-Pyramid-Networks-for-Object-Detection/)的帮助下，FCOS的BPR也达到了相当的水平，为98.40%。由于当前检测器实际的最优recall远低于90%，所以FCOS和[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)在BPR上的这点细小差距不会影响到检测器性能。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/3.png)

>个人注解：因为下采样或者anchor和GT box的匹配机制，有可能导致某些GT box无法分配到任何一个样本上。以FCOS为例，如果有一个GT box，feature map上的任意一点还原到输入图像上都无法落在这个GT box内，则这个GT box就无法分配给任何一个样本了，这也就导致了BPR不是100%。以[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)为例，如果有一个GT box，在一定的IoU阈值下，没有任何一个anchor可以与之匹配，则这个GT box就无法分配给任何一个样本了。表1中第一行的“None”表示不考虑低质量的匹配，即此时的IoU阈值肯定是大于0.4的，在[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)中，正样本的IoU阈值为0.5。

👉**Ambiguous Samples.**

模糊样本的示意见Fig1右，即一个样本被分配多个GT box。在表2中，我们展示了在minival数据集上，所有正样本中模糊样本的比例。此外，我们认为如果一个样本被分配的多个GT box属于同一类别，则这样的模糊样本是不重要的。因为无论这种样本预测哪个GT box，预测都算是正确的，因为类别都是一样的。而被遗漏的其他GT box只能通过别的样本来预测。因此在表2中，我们也统计了去除这种模糊样本后的比例（见“(diff.)”列）。为了进一步证明GT box的重叠对于FCOS来说并不是一个问题，我们统计了在推理阶段有多少检测框是来自模糊位置的。我们发现仅有2.3%的检测框来自模糊位置。如果只考虑不同类别的重叠，则这一比例降低至1.5%。但这并不意味着FCOS在这1.5%的模糊位置上是不能工作的。如之前提到的，这些位置会被分配给面积最小的GT box。因此，对于这些位置来说，只是存在遗漏一些较大目标的风险。如接下来的实验所示，它们并没有使FCOS变得不如anchor-based检测器。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/5.png)

### 4.1.2.With or Without Center-ness

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/6.png)

在表4中，结果基于minival数据集。第一行表示不使用center-ness。第二行表示使用预测的回归向量计算center-ness（不引入额外的center-ness分支）。第三行表示直接使用center-ness分支预测center-ness的值。[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)使用两个IoU阈值来划分正负样本，这也能减少低质量的预测。center-ness的方式就可以省去这两个IoU阈值的超参数。但在论文提交之后，我们发现结合center-ness和IoU阈值可以得到更好的结果，在表3中我们用“+ ctr. sampling”表示。

### 4.1.3.FCOS vs. Anchor-based Detectors

上述FCOS和标准的[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)有两个细微的差别。1）在新加的卷积层中，除了最后的预测层，我们使用了[GN（Group Normalization）](http://shichaoxin.com/2024/08/20/论文阅读-Group-Normalization/)，这让训练更加稳定。2）我们使用$P5$生成$P6,P7$，而没有像[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)那样使用$C5$。我们发现使用$P5$可以稍微提升些性能。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/7.png)

“+ ctr. on reg.”表示将center-ness分支移到回归分支。“Normalization”表示用[FPN](https://shichaoxin.com/2023/12/19/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Feature-Pyramid-Networks-for-Object-Detection/)层级的步长对式(1)进行归一化。

## 4.2.Comparison with State-of-the-art Detectors

基于MS-COCO test-dev数据集，我们比较了FCOS和其他SOTA的目标检测器。对于这些实验，在训练阶段，我们将输入图像的短边随机缩放到640到800之间，并且将迭代次数加倍到180K次（学习率变化的时间点也对应加倍）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/8.png)

据我们所知，这是第一次一个没有任何花哨技巧的anchor-free检测器，其性能远远优于anchor-based检测器。

# 5.Extensions on Region Proposal Networks

使用FCOS代替[Faster R-CNN](https://shichaoxin.com/2022/04/03/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Networks/)中的RPN部分来生成proposal。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/9.png)

# 6.Conclusion

我们提出了一种anchor-free且proposal-free的单阶段检测器FCOS。接下来是附录部分。

# 7.Class-agnostic Precision-recall Curves

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/10.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/11.png)

Fig4-6是在minival数据集上，不同IoU阈值下的与类别无关的[PR曲线](https://shichaoxin.com/2018/12/03/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%89%E8%AF%BE-%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%BA%A6%E9%87%8F/#31p-r%E6%9B%B2%E7%BA%BF)。3个曲线对应的AP见表7。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/12.png)

# 8.Visualization for Center-ness

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/13.png)

# 9.Qualitative

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/14.png)

# 10.More discussions

👉**Center-ness vs. IoUNet:**

和IoUNet的比较，不再详述。

>IoUNet：Acquisition of Localization Confidence for Accurate Object Detection。

👉**BPR in Section 4.1 and ambiguity analysis:**

不再详述。

👉**Additional ablation study:**

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/FCOS/15.png)

👉**RetinaNet with Center-ness:**

center-ness不能直接用于[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)，因为feature map上的一个位置对应一个center-ness，而[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)中，feature map上的一个位置对应多个anchor box，每个anchor box需要不同的center-ness值。

对于[RetinaNet](https://shichaoxin.com/2024/02/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Focal-Loss-for-Dense-Object-Detection/)，anchor box与GT box之间的IoU分数可能可以作为center-ness的替代方案。

👉**Positive samples overlap with RetinaNet:**

我们想强调的是，center-ness只有在推理阶段才发挥作用。在训练阶段，所有落入GT box内的像素点位置都会被标记为正样本。因此，FCOS可以使用更多的前景位置来训练回归器，从而产生更准确的边界框。

# 11.原文链接

👽[FCOS：Fully Convolutional One-Stage Object Detection](https://github.com/x-jeff/AI_Papers/blob/master/2024/FCOS：Fully%20Convolutional%20One-Stage%20Object%20Detection.pdf)