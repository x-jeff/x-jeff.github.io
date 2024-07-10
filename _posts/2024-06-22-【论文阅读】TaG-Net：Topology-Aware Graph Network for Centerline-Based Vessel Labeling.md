---
layout:     post
title:      【论文阅读】TaG-Net：Topology-Aware Graph Network for Centerline-Based Vessel Labeling
subtitle:   TaG-Net，vessel labeling，vessel segmentation
date:       2024-06-22
author:     x-jeff
header-img: blogimg/20220824.jpg
catalog: true
tags:
    - AI Papers
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.INTRODUCTION

血管疾病是全球人类死亡的一个主要原因。CT血管造影（Computed Tomography Angiography，CTA）是检查血管狭窄、阻塞、动脉瘤以及其他血管异常的一个主要技术。在临床实践中，医生必须手动编辑血管分割、校正3D重建错误、标记血管片段，以进行定量诊断，这是非常费时费力的。因此，自动且准确的血管分割和标记是非常有用的。

>个人注解：CTA多用于观察动脉，但也可用于观察静脉。

自动血管分割和标记也是计算机辅助诊断中生成报告的先决条件。在血管分割之后，自动标记单个血管片段以进行解剖定位是非常令人感兴趣的。

头颈部动脉树从主动脉弓（Aortic Arch，AO）一直延伸到Willis环（the circle of Willis，CoW），其可分为18段。下图中从大动脉弓（即AO）到脑动脉环（即Willis环）这一部分就是头颈部动脉树。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/1.png)

其中有13段属于颈部血管：

* 主动脉弓（AO）。
* 头臂动脉（brachiocephalic artery，BCT），也称无名动脉（innominate artery）。
* 颈总动脉（common carotid artery，L/R-CCA），分为左颈总动脉和右颈总动脉。
* 锁骨下动脉（subclavian artery，L/R-SA），分为左锁骨下动脉和右锁骨下动脉。
* 颈内动脉（internal carotid artery，L/R-ICA），分为左颈内动脉和右颈内动脉。
* 颈外动脉（external carotid artery，L/R-ECA），分为左颈外动脉和右颈外动脉。
* 椎动脉（vertebral artery，L/R-VA），分为左椎动脉和右椎动脉。
* 基底动脉（basilar artery，BA）。基底动脉由左右两个椎动脉汇合而成。

有5段属于头部血管：

* 大脑中动脉（middle cerebral artery，L/R-MCA），分左右，是颈内动脉的直接延续，不参与Willis环的组成。
* 大脑后动脉（posterior cerebral artery，L/R-PCA），分左右，起自基底动脉。
* 大脑前动脉（anterior cerebral artery，L/R-ACA）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/2.png)

Fig1展示了头颈部主要血管。我们可以看到，血管横跨整个图像volume，是具有长、曲折和分支的管状结构，具有不同的形状和大小。我们的目标是对18段血管进行标注，如Fig1(c)所示。此外，Fig1(d)所示的血管中心线具有血管的拓扑结构和解剖结构，可作为血管分割和标注的先验知识。我们可以利用中心线对血管进行标注。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/3.png)

## 1.A.Related Works

在本节中，我们首先介绍了有关CTA图像中血管分割和标注的相关工作。然后，我们简要回顾了血管结构的图表示以及点云学习方法。

### 1.A.1.Vessel Segmentation and Labeling

本部分介绍已经提出的用于头颈CTA的血管分割方法。参考文献1在传统骨骼配准、subtraction和区域生长的基础上，结合形态学运算，进行了头颈部血管分割。参考文献4提供了一种用于头颈部CTA重建的CerebralDoc系统，其中应用了3个级联的ResU-Net。此外，还开发了一个连通生长模型来校正中断的血管。在参考文献5中，使用点云网络来refine从[V-Net](https://shichaoxin.com/2023/08/01/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-V-Net-Fully-Convolutional-Neural-Networks-for-Volumetric-Medical-Image-Segmentation/)获得的血管分割结果，并对血管mask进行膨胀操作以防止血管中断。然而，由于基于CNN分割的固有局限性，颈部血管的较长断连区域无法被接上。此外，还有一些早期的方法通常使用有限的FOV，如颈动脉（见参考文献7）、大脑（见参考文献8，9）或主动脉（见参考文献10）区域。

一些文献针对冠状动脉（见参考文献11，12，13，14）、脑动脉（见参考文献3，15，16）以及头颈血管（见参考文献5），提出了解剖学标注算法。一些基于配准的方法（见参考文献12，15，16），其基于解剖结构的先验知识进行标注任务。但由于个体差异较大，通常需要基于规则的refine。参考文献3对MRA脑血管系统进行了联合分割和标注。他们首先构建了一个完整的图来获得脉管系统，然后选择并标注最有可能是真实脉管系统的边的集合。参考文献12提出了一种TreeLab-Net框架来从分割的血管中提取中心线，然后使用2D球面坐标转换对中心线进行投影以提取特征，最后使用双向树结构长短期记忆网络进行标注。

图卷积网络（Graph Convolutional Network，GCN）可以和CNN或点云相结合来学习解剖结构的体素关系。参考文献13通过结合从TreeLab-Net提取的特征、从GCN学到的图信息以及用于冠状动脉解剖标注的3D图像特征，设计了一种条件部分残差图卷积网络（conditional
partial-residual graph convolutional network，CPR-GCN）。参考文献5提出了HaN-Net框架，用于分割头颈部的13根血管，其通过点云网络和单个GCN层相结合的方法来学习体素关系。参考文献14使用GCN训练了一个点云网络，命名为CorLab-Net，用于冠状动脉标注。

然而，参考文献12，13中构建的图只是利用分支作为节点，其血管拓扑结构没有完全保留。参考文献5，14中构建的图是基于初始分割中血管点之间的欧氏距离。这可能会导致边连接错误。这可能会违反血管解剖结构，得到错误的标注结果。

### 1.A.2.Graph Representation of Vascular Structures

图神经网络是一个快速发展的领域，已被广泛应用于医学图像处理（见参考文献17），以研究血管解剖、连接、结构异常，甚至模拟血流（见参考文献18）。对于上述血管标注的工作，有几种方法可以构建血管图并使用GCN来学习图表示。血管图的构建方法可分为：

1. 节点表示分叉点和端点，边表示分支。见参考文献3。
2. 节点表示分支，边表示分叉点。见参考文献12，13。
3. 节点表示从初始分割中选择的点，边表示点之间的欧氏距离。见参考文献5，14。

使用前两种方法构建的图具有合适的解剖结构，但不能捕捉整个血管拓扑结构。最后一种方法可以捕捉整个结构，但可能存在错误连接，导致解剖信息不准确。因此，这些血管图无法准确捕捉到血管的拓扑结构和解剖信息。此外，在GCN学习图表示的过程中，它们的图被固定在一个单一尺度上。其缺乏在不同尺度上捕捉局部上下文的能力。因此，探索构建血管图的方法以及如何利用它更好地捕捉血管的拓扑结构和解剖结构仍然是一个悬而未决的问题。

### 1.A.3.Point Cloud Learning

最近，点云在深度学习中备受关注（见参考文献19）。点云学习方法可分为基于点的（见参考文献20，21，22，23）和基于树的（见参考文献24）。3D点云在分类和分割方面取得了令人印象深刻的成就。特别是，参考文献21提出了PointNet++，其通过设计金字塔特征聚合框架来学习全局和局部特征。参考文献23提出的DGCNN使用一个名为Edge-Conv的模块来学习拓扑信息。参考文献22提出的RS-CNN通过采用层级框架来学习上下文的形状关系（即点之间的几何拓扑约束），用于分析点云。参考文献24提出的Kd-Network以3D点云构建的Kd-Tree为输入，使用具有skip connection的编码器-解码器框架为网络，进行点云分割。

Fig1(d)展示了线空间内的血管中心线，其由沿着血管的中心点组成，并具有树状结构的固有图表示。因此，几何深度学习的重要技术（见参考文献25）可以用于处理图和点云（即中心线点）数据，因为这些数据是在线空间而不是3D体素空间，这有效解决了血管标注问题。

## 1.B.Challenges

与大多数标注工作类似，我们先进行血管分割，然后根据分割结果进行血管标注。然而，这样的pipeline仍然存在局限性和挑战。Fig2的上半部分展示了血管分割面临的难点，Fig2的下半部分是在体素空间中使用CNN（比如参考文献26中的nnU-Net）进行血管标注所面临的问题。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/4.png)

在Fig2的上半部分，我们展示了具有粘连或中断问题的三个错误预测例子。第一个例子中，两个独立的血管被粘连在一起，是因为它们在空间上很近。第二个例子中，血管中断是由于弯曲结构造成的。第三个例子中，因为图像质量不佳，造成了血管中断。

在Fig2的下半部分，展示了5个血管标注错误的例子。这些例子中的一个常见错误是空间混乱（spatial confusion）。当不同的血管段在空间上接近时，可能会发生空间混乱。这是因为CNN通常在欧几里得空间中平等地对待每个体素，并且缺乏处理解剖结构中个体变异性的能力。因此，血管标注会受到其空间相邻东西的影响。如果不考虑解剖和拓扑结构，很难纠正这些错误。

空间混乱可能会导致两种形式的错误。一种是前三列所展示的解剖结构不正确。另一种是后两列中血管标注不正确。不正确的分割也可能导致空间混乱。第一列是分割粘连造成的空间混乱。第二列是分割中断造成的空间混乱。

总的来说，在体素空间中进行头颈部血管分割和标注的主要挑战包括：

* 在血管分割中，通常由于不同程度的血管弯曲、周围组织的可变性、紧密分布的血管以及有限的图像质量，会造成血管中断或粘连。
* 由于解剖结构的个体差异、空间上接近但解剖结构上独立、血管分割的中断和粘连，导致了血管标注出现空间混乱。

## 1.C.Contributions

我们不是在体素空间中进行处理，而是在线空间中构建血管图，并使用拓扑和解剖信息来指导血管标注。我们进一步利用线空间和体素空间之间的相互作用来改进血管标注和分割的正确性。我们的贡献总结如下：

* 为了解决体素空间中血管标注的空间混乱问题，我们在线空间中构建了中心线血管图，并提出了一种新的拓扑感知图网络（**T**opology-**a**ware **G**raph **Net**work，TaG-Net）用于中心线标注。
* 为了缓解血管分割的中断和粘连问题，我们利用TaG-Net标注的血管图来改进体素空间中的血管分割。
* 我们在401组CTA扫描数据上验证了我们所提出的头颈部血管分割和标注方法。大量实验表明，我们的方法优于其他SOTA方法。

# 2.METHOD

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/5.png)

我们所提出的方法的框架见Fig3。首先，我们从初始血管分割中提取中心线，然后用中心线构建血管图。其次，我们使用TaG-Net来对中心线血管图的拓扑结构和解剖特征进行编码，并将中心线标注为不同的片段。最后，我们使用一个基于已标注血管图的血管补全步骤来解决初始分割中的中断和粘连。同时，将中心线的标注分配给refine后的血管分割，得到血管标注的结果。

## 2.A.Centerline Extraction and Graph Representation

如Fig1(d)所示，头颈部血管中心线具有用于血管结构的树状拓扑的固有图表示。因此，为了缓解体素空间中的空间混乱，并利用血管的拓扑和解剖结构，我们基于中心线构建血管图，并在线空间中进行血管标注。

首先，先单独训练一个分割网络（SegNet），用于在体素空间中生成初始血管分割$Y_V$。在本研究中，一个级联的nnU-Net（见参考文献26）用于从CTA图像中分割血管，因为它可以处理各种目标结构，并为医学图像分割产生令人满意的结果。一个传统的3D细化算法（见参考文献27）被用于提取中心线。

然后，使用提取到的中心线构建血管图，其节点由中心线点形成，边由它们的连通性表示。具体来说，从中心线构建树结构使用了经典的最近邻搜索方法Kd-Tree（见参考文献28）。在构建树之后，我们保留长度短于$r$的边，并用这些边构建初始血管图。然后，采用后处理来去除像闭环三角形一样的冗余边。最后，一个拓扑感知的血管图$G_L$被构建完成，其有$N_L$个节点。

需要注意的是，参考文献5中构建的图使用从初始分割中选出的点作为节点，这些点覆盖整个血管。边是基于欧氏距离（带阈值）来构建的。因此，它可能在空间上接近但解剖学上独立的血管之间存在错误连接，这会导致空间混乱。相比之下，本文构建的中心线血管图更准确，可以更好的表示血管的树状结构。

## 2.B.Centerline Labeling Using TaG-Net

基于上述获得的中心线血管图，使用TaG-Net将中心线标记为18个片段。如Fig4所示，TaG-Net由编码器和解码器组成。有4个级别的SA（set abstraction）模块用于学习中心线特征和图表示，有4个FP（feature propagation）模块用于传播特征以预测每个点的标签。它在几何深度学习中同时利用了点云网络的点处理和GCN的图表示。也就是说，点云网络处理中心线的空间关系和拓扑几何结构。GCN学习血管图的表示。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/6.png)

### 2.B.1.Hierarchical Features Learning

我们引入了一个层级的特征学习框架来捕捉不同尺度下血管图的局部上下文。它由四个SA模块组成，用于处理多尺度血管图，并沿着层级结构逐步抽象出越来越大的局部区域。在每个模块中，对一组节点和相应的血管图进行处理和抽象，以生成具有较少节点的新血管图。

每个SA模块中有4种类型的运算符：

1. 拓扑保持采样：topology-preserving sampling（TPS）
2. 拓扑感知特征分组：topology-aware features grouping（TFG）
3. PointNet
4. GCN

TPS对输入的血管图进行下采样。下采样图的节点是局部区域的质心。这些质心及其构成的血管图保留了血管的拓扑和解剖结构。TFG基于TPS选择的质心构建局部血管区域。PointNet（见参考文献20）用于对这些局部区域的节点特征进行编码。GCN（见参考文献29）直接在血管图上运行，以对图结构中的节点特征进行编码。

如Fig4左下部分所示，SA模块的输入有两部分，一部分是一个$N \times C$的矩阵，其中，$N$表示节点，$C$表示每个节点对应的特征向量维度；另一部分是一个[邻接矩阵](https://shichaoxin.com/2023/05/15/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E4%BA%94%E7%AB%A0-%E5%9B%BE%E7%9A%84%E9%81%8D%E5%8E%86/)$A$，大小为$N \times N$。输出对应也是两部分，一部分是一个$N' \times C'$的矩阵，其中，$N'$是下采样后的节点，$C'$是每个下采样节点对应的新特征向量的维度；另一部分是一个新的[邻接矩阵](https://shichaoxin.com/2023/05/15/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E4%BA%94%E7%AB%A0-%E5%9B%BE%E7%9A%84%E9%81%8D%E5%8E%86/)$A'$，大小为$N' \times N'$。

#### 2.B.1.a.Topology-preserving sampling (TPS)

TPS的目标是对图进行下采样，同时根据血管的拓扑结构保持节点之间的空间关系。如Fig5所示，可以从上一个图中采样得到不同节点数的血管图，并保留其血管拓扑结构。血管图的4个尺度对应SA模块的4个级别。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/7.png)

具体来说，不同位置的节点对血管图的解剖结构和拓扑结构的贡献也不同。因此，我们将节点分为关键节点和普通节点。节点的重要性取决于该节点分支出去的血管数。我们将血管图$G_L$中度不等于2的节点视为关键节点（即分叉点、连接点和端点）。

如Fig4左下部分所示，我们从$N$个节点下采样到$N'$个节点。在采样过程中，为了保持血管的拓扑和解剖结构，$N_k$个关键节点会被保留。也就是说，我们只对普通节点进行下采样，即$N-N_k$个普通节点被下采样到$N'-N_k$个，下采样方法为最远点采样（farthest point sampling，见参考文献21）。因此，$N'$个节点包括所有的关键节点和被采样后的普通节点。最后，我们将[邻接矩阵](https://shichaoxin.com/2023/05/15/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E4%BA%94%E7%AB%A0-%E5%9B%BE%E7%9A%84%E9%81%8D%E5%8E%86/)$A$更新为$A'$，以建立这些新节点之间的连接。

#### 2.B.1.b.Topology-aware features grouping (TFG)

TFG将$N'$个节点视为质心，进行用于特征聚合的分组操作。对于每个质心，我们采用球查询方法（ball query method，见参考文献21）来获得相应的局部区域，如Fig4中的紫色球所示。我们没有像参考文献21中那样对局部区域内的所有节点进行分组，我们仅对在局部区域内和质心有连接的$K$个节点进行分组，如Fig4中的紫色血管路径所示。因此，TFG的输出是一个大小为$N' \times K \times C$的矩阵，$N'$表示有$N'$个组，每个组内有$K$个节点，每个节点的特征维度为$C$。

我们提出的TFG确保了信息仅在具有适当拓扑和解剖结构的节点之间传递。它防止了空间上接近但解剖学上独立的血管之间的影响。

#### 2.B.1.c.The pointnet and GCN layers

这两层用来对节点特征进行编码。PointNet层使用$N'$个局部区域作为输入，即输入大小为$N' \times K \times C$。PointNet使用$K$个节点对其质心进行基于局部区域的编码。GCN层的输入为一个$N' \times C$的矩阵和[邻接矩阵](https://shichaoxin.com/2023/05/15/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E4%BA%94%E7%AB%A0-%E5%9B%BE%E7%9A%84%E9%81%8D%E5%8E%86/)$A'$。因此，SA模块的输出大小为$N' \times C'$。新的节点特征的维度为$C'$，新的节点特征concat了PointNet学到的节点特征和GCN学到的节点特征。

### 2.B.2.Feature Propagation for Centerline Labeling

FP模块用于解码特征。如Fig4所示，采用层级传播的策略来获得原始节点的新节点特征。FP模块由两层构成：插值层和unit pointnet层。插值层将$N'$个节点恢复到$N$个节点。使用基于血管图上的3个邻居节点（拓扑结构上的相邻）的逆距离加权平均来获得插值节点。unit pointnet层用于更新每个节点的特征向量。插值后的$N$个节点会和同级别SA模块通过skip concatenation将特征concat在一起。因此，unit pointnet层的输入大小为$N \times (C' + C)$。然后，每个节点的特征向量会被更新，新的特征向量维度为$C''$。FP过程会一直重复执行，直到特征被传播到原始的$N_L$个节点。在最后一个FP模块之后，全连接层将每个节点分配给18个标签中的一个。

## 2.C.Vessel Completion and Labeling

上述过程得到了初始血管分割$Y_V$和相应的标注中心线血管图$S_L$，其中线空间的中心线可视为是体素空间中血管分割的high-level表示。我们可以基于$S_L$进行中心线补全，从而进一步补全血管，以解决中断问题。此外，可以将中心线的标注分配给邻近的体素来获得体素空间的血管标注。注意，本章节的refine工作仅在推理阶段进行，不需要训练。

### 2.C.1.Centerline Completion

可以使用基于拓扑和解剖结构的先验知识来补全有中断的$S_L$。如Fig6所示，我们首先进行标签内的补全（intra-label completion），然后进行标签间的补全（inter-label completion），最后得到refine后的标注中心线。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/8.png)

如Fig6(b)所示的标签内补全，先搜索连接对，然后将具有相同标签的连接对连接在一起。我们首先找到$S_L$中度为1的节点，因为它们位于分段的开始/结束处。然后，我们对这些节点进行配对，并计算这些节点对的测地线距离（geodesic distance）。需要注意的是，来自同一分段的节点被排除在外。然后，具有最小距离的节点对被认为是两个相邻分段的连接对。重复上述过程，直到我们获得所有的连接对，将这些分段连接成一个完整的血管。

>个人注解：测地线距离是指在曲面或更一般的流形上的两点之间沿着表面的最短路径的距离。
>
>* 在欧几里得空间中（如平面或三维空间），测地线与直线等价。因此，测地线距离就是两点之间的直线距离。
>* 在曲面（例如球面）上，测地线通常是曲线。例如，在地球表面上，测地线是两点之间的大圆弧。这种路径通常与简单的直线不同，因为它考虑到了表面的曲率。
>* 在图论中，测地线距离是指两个节点之间的最短路径上的边的数量或权重总和。

对于Fig6(a)所示的标签间补全，我们利用了头颈部解剖结构的先验知识。首先，基于$S_L$找到已经连接的且具有不同标签的标签对。拿这些标签对和标准的解剖图进行比较，得到缺失的标签对。然后我们需要找到缺失标签对的连接对。给定一个缺失的标签对，我们可以找到具有这两个不同标签的血管分段的起始/结束节点。然后可以采用和标签内补全类似的方式获得标签间连接对。

在得到标签内和标签间的连接对后，使用[Dijkstra算法](https://shichaoxin.com/2023/07/11/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E5%85%AD%E7%AB%A0-%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84/#2dijkstra%E7%AE%97%E6%B3%95%E9%80%9A%E8%BF%87%E8%BE%B9%E5%AE%9E%E7%8E%B0%E6%9D%BE%E5%BC%9B)（见参考文献30）在局部原始图像上搜索连接对节点之间的最短路径。这个最短路径便被认为是连接路径，如Fig6(a)和Fig6(b)所示。

可以基于解剖结构去除$S_L$中的粘连。当具有不同标签的血管段之间存在粘连时，我们可以根据先验的解剖结构在已标注的血管图上定位到错误连接。然后，我们通过切断图上的连接并移除冗余的分支来移除粘连路径。

>本部分的源码详解：[【源码解析】VesselCompletion in Tag-Net](http://shichaoxin.com/2024/07/09/源码解析-VesselCompletion-in-Tag-Net/)。

### 2.C.2.Vessel Completion

根据中断区域的连接路径，并结合初始分割$Y_V$和原始图像，我们通过区域生长（见参考文献31）补全初始分割。连接路径上的点被视为种子点。同时，利用与目标标签相对应的初始分割部分的平均半径作为约束来保持管状形状。当生长区域的半径大于该目标标签对应的初始分割部分的最大半径时，我们根据该标签的平均半径来去除多余区域。类似的方法，根据上述提到的粘连路径来对粘连进行去除。

### 2.C.3.Vessel Labeling

在获得refine后的标注中心线和refine后的分割之后，我们基于有向距离图（directional distance maps）将refine后的标注中心线的标签分配给距离最近的refine后的分割上，以生成标注的mask $S_V$。我们会计算每个标注中心线的距离图，使得refine后的分割中的每个体素都能得到其到18个标签的距离，即每个体素对应18个距离。简单来说，距离最短的标签可以分配给这个体素。但是，由于血管段的半径变化较大，这样分配很容易导致错误。具体来说，对于半径较大的血管段，如果它靠近另一个半径较小的血管段，则半径较大的血管段会被错误的分配成较小血管段的标签。因此，使用有向距离图来解决这个问题。它会把血管之间的间隙（通常为背景）赋予很大的距离，这会促使标注的血管在方向上保持一致，这类似于梯度矢量流（见参考文献32）。最终，我们得到了如Fig3所示的准确的标注mask $S_V$。

## 2.D.Implementation Details

我们对SegNet和TaG-Net进行了相同的五折交叉验证，在每一折中，两个模型的训练和测试样本的划分是一致的。首先，使用血管mask的GT和[Dice loss](https://shichaoxin.com/2023/08/01/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-V-Net-Fully-Convolutional-Neural-Networks-for-Volumetric-Medical-Image-Segmentation/)训练SegNet。然后，使用中心线标签的GT和交叉熵损失训练TaG-Net。对于TaG-Net的训练，输入是从血管mask的GT中提取出来的中心线和依据该中心线构建的血管图。对于TaG-Net的推理，输入是从SegNet获得的初始血管分割中提取的中心线和相应的血管图。

用于构建初始血管图的距离阈值$r$根据经验设置为1.75个体素。较小的$r$会导致更多的孤立节点和中断，而较大的$r$则会导致更多的冗余边。在TPS中，基于我们的数据集，输入血管图$G_L$的关键节点数量平均约为350个，所以我们将血管图节点的最小数量设置为512。如Fig5所示，最后一个血管图有512个节点，可以保留整个血管的拓扑结构。在训练和测试阶段，点坐标和对应的半径都进行了归一化处理。在TFG过程中，4个SA模块的球查询方法的半径分别设置为0.1、0.2、0.4和0.8。半径过小可能会导致局部区域受限，而半径过大则可能和其他血管段产生混淆，并使网络难以进行标注。

在我们的实验中，所有的分割结果都是根据中心线感知的噪声去除来进行refine的。与用于去除小冗余区域的基于连接部件面积的方法相比，基于中心线长度的方法适用于具有特殊管状结构的血管，该结构在头部血管中具有较小的半径，而在颈部血管中具有较大的半径。用于去除小区域的中心线分段长度的阈值被设置为15个体素。所有实验都使用PyTorch，运行在2块NVIDIA Titan RTX GPU上。

# 3.EXPERIMENTS

本章节报告了实验结果，以验证我们提出的方法。代码开源地址：[TaG-Net](https://github.com/PRESENT-Y/TaG-Net)。

## 3.A.Experimental Setup

### 3.A.1.Materials

数据是我们在四川大学华西医院收集的内部CTA图像（包含头颈部），伦理审查委员会已经审查并批准了这项研究。一共收集了来自不同患者的401组数据，axial size都是$512 \times 512$。intra-slice spacing的范围从0.361mm到0.707mm，inter-slice spacing的范围从0.4mm到0.7mm。数据的层数基本在400到900之间。

>个人注解：
>
>* intra-slice spacing：指的是在同一CT切片（即单个图像平面）内相邻像素（或体素）之间的实际物理距离，通常以毫米表示。也就是我们常说的FOV。
>* inter-slice spacing：指的是相邻的CT切片（即连续图像平面）之间的实际物理距离，通常以毫米表示。也就是我们常说的z-spacing，即z方向的分辨率。

放射科医生使用Mimics软件对头颈部血管的标签进行标注，并由高级放射科医生进行审查，这些标注作为GT。手动标记一个病例大约需要3个小时。

### 3.A.2.Metrics

对于血管标注任务，我们用每个血管段的DSC（Dice similarity coefficient）、precision和recall来评估标注结果。此外，我们还计算了95HD（95% Hausdorff distance，见参考文献33），因为它对异常值很敏感。

对于血管分割任务，我们通过计算volume-wise的指标来评估分割结果：DSC、95HD和ASD（average Surface distance，见参考文献33）。此外，还使用了中心线指标和拓扑连接性指标。中心线指标包括正确性、完整性和质量。正确性指的是正确提取的血管中心线的百分比。完整性指的是和GT匹配的中心线长度占GT中心线总长度的比率。使用质量这一指标评估提取到的中心线的好坏程度。它会考虑中心线的完整性和正确性。这些指标的计算公式见下：

$$corr. = \frac{\lvert \hat{Y}_{cl} \bigcap Y \rvert}{\lvert \hat{Y}_{cl} \rvert} \tag{1}$$

$$compl. = \frac{\lvert Y_{cl} \bigcap \hat{Y} \rvert}{\lvert Y_{cl} \rvert}\tag{2}$$

$$qual. = \frac{\lvert Y_{cl} \bigcap \hat{Y} \rvert + \lvert \hat{Y}_{cl} \bigcap Y \rvert}{\lvert Y_{cl} \rvert + \lvert \hat{Y}_{cl} \rvert} \tag{3}$$

其中，$corr., compl., qual.$分别表示正确性、完整性和质量。分割GT为$Y$，从中提取到的中心线表示为$Y_{cl}$，分割预测结果表示为$\hat{Y}$，从中提取到的中心线表示为$\hat{Y}_{cl}$。拓扑精度指标使用Betti误差（见参考文献34）。Betti误差直接比较分割预测结果和GT之间的handle数量。

## 3.B.Quantitative Results

### 3.B.1.Vessel Labeling

为了评估血管标注的性能，我们将TaG-Net和HaN-Net（见参考文献5）、nnU-Net（见参考文献26）进行了比较。对于每段血管，DSC、95HD、precision、recall都是根据GT计算的。由于颈部血管的半径大于头部血管的半径，因此分别计算头部和颈部血管的指标平均值。HaN-Net最初的版本只能标注13根血管，为了公平的比较，我们将其重新实现，扩展到18根血管。具体来说，就是添加了R/L-SA和R/L-ECA，并将PCA分为R/L-PCA。

通过[配对t检验](https://shichaoxin.com/2019/01/30/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E5%9B%9B%E8%AF%BE-%E7%BB%9F%E8%AE%A1%E5%AD%A6%E7%9F%A5%E8%AF%86%E4%B9%8B%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C/#32%E9%85%8D%E5%AF%B9%E6%A0%B7%E6%9C%ACt%E6%A3%80%E9%AA%8C)比较了上述3种方法在每段血管上的DSC和95HD，此外，还比较了在头部或颈部整个血管段上的平均指标。显著性水平设为0.05。如表1所示，TaG-Net在所有血管上的平均性能见最后一行：$DSC=91.5\%, 95HD=3.600mm, PRE=93.0\%, REC=90.2\%$，相比HaN-Net，性能分别提升了$1.9\%, 1.089mm, 1.4\%, 2.4\%$。相比基于CNN的标注方法（即nnU-Net），提升更为明显。总的来说，TaG-Net在我们的数据集上取得了最好的结果。统计分析的结果也表明，与HaN-Net相比，除了AO和R-ICA的DSC外，其余指标的$p$值都低于0.05。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/9.png)

如果关注头部血管的平均指标，相比HaN-Net，TaG-Net的DSC提升3.6%、95HD提升0.457mm、PRE提升1.8%、REC提升5.4%。如果关注颈部血管的平均指标，DSC的提升为1.3%、95HD的提升为1.335mm、PRE的提升为1.2%、REC的提升为1.2%。事实上，我们只对颈部血管进行了后处理（血管补全和粘连去除）。因此，头部结果的提升源自TaG-Net基于中心线的血管标注。而颈部结果的提升得益于TaG-Net和基于标注血管图的后处理。

从结果的细节来看，TaG-Net对颈部血管的性能比头部血管更好，这可能是因为颈部血管的尺寸相对较大且结构不复杂。对于颈部血管，AO取得了最高的DSC（97.7%），而BA的DSC最低（88.1%）。另一方面，R-SA和L-SA的95HD比较大，这可能是因为分割预测结果和GT的结束位置不同造成的。一样的情形也发生在R-ECA和L-ECA。对于BCT和BA，分别连接AO和R-SA、VA和PCA，它们的小尺寸以及位置也会影响标注性能。在头部分割中，它们在DSC指标上实现了类似的性能，但在95HD指标上略有不同。

### 3.B.2.Vessel Segmentation

为了验证在血管分割上的性能，我们将我们提出的方法和一些流行的基于CNN的方法进行了比较，比如[3-D U-Net](https://shichaoxin.com/2023/07/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-3D-U-Net-Learning-Dense-Volumetric-Segmentation-from-Sparse-Annotation/)、[V-Net](https://shichaoxin.com/2023/08/01/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-V-Net-Fully-Convolutional-Neural-Networks-for-Volumetric-Medical-Image-Segmentation/)以及nnU-Net（见参考文献26）。我们也提供了参考文献4，7，8，10，9中方法的结果，其中，参考文献4和我们分割了一样的血管，其他的参考文献7，8，9，10只是分割了头颈血管的一部分。此外，我们也提供了参考文献5的血管分割结果。

表2报告了血管分割、中心线以及血管连通性的比较结果。血管分割的评估指标使用了DSC、95HD和ASD，中心线评估指标使用了$qual.$，连通性指标使用Betti误差。我们的方法使用nnU-Net作为初始分割，从表2可以看到，DSC提升了1.2%（95.9% vs. 94.7%）。相比其他方法，我们的方法“nnU-Net refined by TaG-Net”取得了最好的结果。和参考文献4的方法（测试数据一共包含18766组头颈部CTA扫描）相比，我们的方法在DSC上略有提升（95.9% vs. 95.1%）。但是，参考文献4的方法没有进行血管标注。最后，[配对t检验](https://shichaoxin.com/2019/01/30/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E5%9B%9B%E8%AF%BE-%E7%BB%9F%E8%AE%A1%E5%AD%A6%E7%9F%A5%E8%AF%86%E4%B9%8B%E5%81%87%E8%AE%BE%E6%A3%80%E9%AA%8C/#32%E9%85%8D%E5%AF%B9%E6%A0%B7%E6%9C%ACt%E6%A3%80%E9%AA%8C)也显示我们的方法在DSC、95HD、ASD、$qual.$和Betti误差等指标上优于其他方法。所有的$p$值都低于0.05。定量结果表明，我们提出的方法达到了SOTA。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/10.png)

### 3.B.3.Ablation Study

我们还对用于中心线标注的TaG-Net进行了消融实验，分别测试去掉GCN、TFG、TPS后的结果。实验结果见表3。如果移除GCN模型，性能下降了1.1%（比如，F1-score=97.7% vs. 96.6%）。如果再移除TFG，性能会进一步下降（96.6% vs. 94.9%）。如果继续移除TPS，F1-score从94.9%下降到了94.5%。消融实验的定量结果表明，TPS、TFG和GCN可以提高标注性能。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/11.png)

## 3.C.Qualitative Results

4名患者（a,b,c,d）的血管分割和标注的定性结果见Fig7。每一行代表一个患者。从左到右分别是nnU-Net、HaN-Net、TaG-Net以及GT。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/TaGNet/12.png)

### 3.C.1.Vessel Labeling

如Fig7所示，可以观察到两种形式的空间混乱：

1. $a_1$①（下）、$b_1$①和$d_1$①展示了nnU-Net结果中的一些不正确的解剖结构。
2. 表面的空间混乱可见于$a_1$①（上）、$a_2$①和$d_2$①。

标注结果中的这些空间混乱经常违反头颈部血管的解剖结构。而我们提出的TaG-Net，见$a_3$、$b_3$和$d_3$，则很好的解决了空间混乱的问题。

标注结果可能会受到图构建方法的影响。nnU-Net直接标注血管，而没有使用图。在HaN-Net中，点云图的构建是基于初始血管分割的所有点（点云）。在TaG-Net中，中心线血管图是由中心线形成的，可以更好地表示血管点之间的空间关系。这从Fig7中可以看出。举个例子，$d_1$①的错误标注并没有被$d_2$①纠正，但在$d_3$中被解决。比较$a_1$③、$a_2$③和$a_3$，可以看出我们解决了由于相邻血管粘连引起的空间混乱。比较$d_1$②、$d_2$②和$d_3$，可以看出我们解决了血管断连的问题。定性结果表明，我们提出的方法沿着同一血管段可以产生一致的标注，还可以去除粘连，解决断连。

与GT相比（第3列和第4列），TaG-Net生成了相似的标注结果，但在不同血管段的连接处仍有一些差异。准确的解剖标注结果表明，我们提出的TaG-Net可以实现令人满意的血管标注性能。此外，TaG-Net只需要标注线空间中的中心线。这证明了我们提出的基于中心线的血管标注方法的有效性。

### 3.C.2.Vessel Segmentation

最终的分割也可以从Fig7中进行可视化评估。将HaN-Net和nnU-Net的分割结果进行比较，HaN-Net对头部血管效果良好，但对颈部血管效果不佳。因此，仅对颈部血管进行补全。

针对颈部血管，我们提出的方法解决了断连问题，可见$a_1$②、$a_2$②和$a_3$，$c_1$②、$c_2$②和$c_3$，$d_1$②、$d_2$②和$d_3$。这是因为标注的血管图提供了准确的拓扑结构和解剖信息来进行血管补全。此外，[Dijkstra算法](https://shichaoxin.com/2023/07/11/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E5%85%AD%E7%AB%A0-%E6%9C%80%E7%9F%AD%E8%B7%AF%E5%BE%84/#2dijkstra%E7%AE%97%E6%B3%95%E9%80%9A%E8%BF%87%E8%BE%B9%E5%AE%9E%E7%8E%B0%E6%9D%BE%E5%BC%9B)提供了完整的中心线，这有助于补全断连的血管。此外，由血管疾病（如血管阻塞、狭窄）引起的中断则不会被补全。例如，病变区域不连接，这和GT保持一致，可见$b_3$②和$b_4$②，$c_3$②和$c_4$②。在我们的结果中，除了血管补全，还可以去除粘连，可见$a_1$③、$a_2$③和$a_3$。

总之，从TaG-Net获得的标注血管图可以改善血管分割，缓解中断和粘连问题。

# 4.DISCUSSION AND CONCLUSION

在本文中，我们提出了一个用于线空间中18根血管的中心线标注的TaG-Net框架，以解决体素空间中血管标注的空间混乱问题。此外，我们通过血管补全改进了体素空间中的分割结果，该方法基于来自TaG-Net的标注血管图，并缓解了血管中断和粘连问题。通过将中心线标签分配给refine后的血管分割来获得最终的标注结果。

我们利用并证明了拓扑感知中心线血管图在头颈部血管标注中的重要性。应用点云网络和图卷积网络的几何深度学习来处理中心线和血管图。为了充分利用头颈部血管的拓扑和解剖信息，提出了拓扑保持采样、拓扑感知分组和多尺度血管图策略。通过这种方式，我们不仅获得了具有正确拓扑结构和解剖结构的准确血管标注结果，而且反过来利用这些拓扑结构和解剖学信息来改进血管分割。实验结果表明，我们的方法达到了SOTA的性能。注意，除了CTA，我们提出的方法还可以应用于其他树结构标注任务，因为它们具有相似的几何特性。

与我们之前的工作HaN-Net相比，本文包括几个显著的改进：

* **Vascular graph construction.** 在本文中，血管结构的拓扑和解剖结构可以更好地用中心线血管图来表示，而之前的工作使用点云来对初始分割进行建模。
* **Network aggregation.** 我们在TaG-Net中提出了TPS和TFG来对节点进行采样并对其特征进行分组，其中多尺度血管图可以保留血管的拓扑和解剖结构。此外，与先前工作中的单个尺度相比，在多尺度血管图上使用GCN来捕捉不同尺度的局部上下文。
* **Improving segmentation with labeled vascular graph.** 标注的血管图不仅使我们得到了更好的血管标注结果，也被用于提升分割结果。
* **Larger dataset and extensive experiments.** 这项工作使用更大的数据集进行了广泛的实验。

尽管有上述创新和优势，但我们提出的方法仍有一些局限性，可以在未来的研究中加以解决。首先，血管补全的方法仅适用于颈部血管。该方法对头部血管的改进并不显著，这可能是因为与颈部血管相比，头部血管更薄且有许多分支结构。这使得很难准确的定位用于搜索连接路径的连接对。将在未来的工作中研究可用于头部血管的具体方法。其次，可以进一步优化为血管mask分配中心线标签的方法，因为我们使用的距离图增加了相当大的计算复杂度。

未来的研究方向包括通过更好地利用中心线血管图的拓扑先验知识，将拓扑感知扩展到端到端的血管分割框架，以解决血管中断和粘连问题。

# 5.REFERENCES

1. A. Hedblom, “Blood vessel segmentation for neck and head computed tomography angiography,” M.Sc. thesis, Linköping Univ., Linköping, Sweden, 2013. [Online]. Available: https://pdfs.semanticscholar.org/28bc/5411955847afee6e3878b35485cdfd0bea84.pdf and https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0237092&type=printable
2. A. Bouthillier, H. R. van Loveren, and J. T. Keller, “Segments of the internal carotid artery: A new classification,” Neurosurgery, vol. 38, no. 3, pp. 425–433, Mar. 1996.
3. D. Robben et al., “Simultaneous segmentation and anatomical labeling of the cerebral vasculature,” Med. Image Anal., vol. 32, pp. 201–215, Aug. 2016.
4. F. Fu et al., “Rapid vessel segmentation and reconstruction of head and neck angiograms using 3D convolutional neural network,” Nature Commun., vol. 11, no. 1, p. 4829, 2020.
5. L. Yao et al., “Graph convolutional network based point cloud for head and neck vessel labeling,” in Machine Learning in Medical Imaging (Lecture Notes in Computer Science), vol 12436, M. Liu, P. Yan, C. Lian, and X. Cao, Eds. Cham, Switzerland: Springer, 2020, doi: 10.1007/978-3-030-59861-7_48.
6. F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in Proc. 4th Int. Conf. 3D Vis. (3DV), Oct. 2016, pp. 565–571.
7. O. Cuisenaire, S. Virmani, M. E. Olszewski, and R. Ardon, “Fully automated segmentation of carotid and vertebral arteries from contrastenhanced CTA,” Proc. SPIE, vol. 6914, Mar. 2008, Art. no. 69143R.
8. D. Babin, A. Pižurica, J. De Vylder, E. Vansteenkiste, and W. Philips, “Brain blood vessel segmentation using line-shaped profiles,” Phys. Med. Biol., vol. 58, no. 22, p. 8041, 2013.
9. M. Livne et al., “A U-Net deep learning framework for high performance vessel segmentation in patients with cerebrovascular disease,” Frontiers Neurosci., vol. 13, p. 97, Feb. 2019.
10. D. Xiaojie, S. Meichen, W. Jianming, Z. He, and C. Dandan, “Segmentation of the aortic dissection from CT images based on spatial continuity prior model,” in Proc. 8th Int. Conf. Inf. Technol. Med. Educ. (ITME), Dec. 2016, pp. 275–280.
11. Q. Cao et al., “Automatic identification of coronary tree anatomy in coronary computed tomography angiography,” Int. J. Cardiovascular Imag., vol. 33, no. 11, pp. 1809–1819, 2017.
12. D. Wu et al., “Automated anatomical labeling of coronary arteries via bidirectional tree LSTMs,” Int. J. Comput. Assist. Radiol. Surg., vol. 14， no. 2, pp. 271–280, 2019.
13. H. Yang, X. Zhen, Y. Chi, L. Zhang, and X.-S. Hua, “CPRGCN: Conditional partial-residual graph convolutional network in automated anatomical labeling of coronary arteries,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020, pp. 3803–3811.
14. X. Zhang, Z. Cui, J. Feng, Y. Song, D. Wu, and D. Shen, “CorLab-Net: Anatomical dependency-aware point-cloud learning for automatic labeling of coronary arteries,” in Machine Learning in Medical Imaging (Lecture Notes in Computer Science), vol 12966, C. Lian, X. Cao, I. Rekik, X. Xu, and P. Yan, Eds. Cham, Switzerland: Springer, 2021, doi: 10.1007/978-3-030-87589-3_59.
15. H. Bogunovic, J. M. Pozo, R. Cardenes, L. San Roman, and A. F. Frangi, “Anatomical labeling of the circle of Willis using maximum a posteriori probability estimation,” IEEE Trans. Med. Imag., vol. 32, no. 9, pp. 1587–1599, Sep. 2013.
16. M. Shen et al., “Automatic cerebral artery system labeling using registration and key points tracking,” in Knowledge Science, Engineering and Management (Lecture Notes in Computer Science), vol. 12274, G. Li, H. Shen, Y. Yuan, H. Wang, H. Liu, and X. Zhao, Eds. Cham, Switzerland: Springer, 2020, doi: 10.1007/978-3-030-55130-8_31.
17. D. Ahmedt-Aristizabal, M. A. Armin, S. Denman, C. Fookes, and L. Petersson, “Graph-based deep learning for medical diagnosis and analysis: Past, present and future,” Sensors, vol. 21, no. 14, p. 4758, Jul. 2021.
18. J. C. Paetzold et al., “Whole brain vessel graphs: A dataset and benchmark for graph learning and neuroscience,” in Proc. 35th Conf. Neural Inf. Process. Syst. Datasets Benchmarks Track (Round), 2021, pp. 1–13.
19. W. Liu, J. Sun, W. Li, T. Hu, and P. Wang, “Deep learning on point clouds and its application: A survey,” Sensors, vol. 19, no. 19, p. 4188, Sep. 2019.
20. R. Q. Charles, H. Su, M. Kaichun, and L. J. Guibas, “PointNet: Deep learning on point sets for 3D classification and segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017, pp. 652–660.
21. C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep hierarchical feature learning on point sets in a metric space,” in Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 1–10.
22. Y. Liu, B. Fan, S. Xiang, and C. Pan, “Relation-shape convolutional neural network for point cloud analysis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2019, pp. 8895–8904.
23. Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, “Dynamic graph CNN for learning on point clouds,” ACM Trans. Graph., vol. 38, no. 5, pp. 1–12, 2019.
24. R. Klokov and V. Lempitsky, “Escape from cells: Deep Kd-networks for the recognition of 3D point cloud models,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Oct. 2017, pp. 863–872.
25. M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst,“Geometric deep learning: Going beyond Euclidean data,” IEEE Signal Process. Mag., vol. 34, no. 4, pp. 18–42, Jul. 2017.
26. F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211,Dec. 2020.
27. T. C. Lee, R. L. Kashyap, and C. N. Chu, “Building skeleton models via 3-D medial surface axis thinning algorithms,” CVGIP: Graph. Models Image Process., vol. 56, no. 6, pp. 462–478, 1994.
28. S. Maneewongvatana and D. M. Mount, “Analysis of approximate nearest neighbor searching with clustered point sets,” 1999, arXiv:cs/9901013.
29. T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” 2016, arXiv:1609.02907.
30. E. W. Dijkstra, “A note on two problems in connexion with graphs,” Numerische Mathematik, vol. 1, no. 1, pp. 269–271, Dec. 1959.
31. R. Adams and L. Bischof, “Seeded region growing,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 16, no. 6, pp. 641–647, Jun. 1994.
32. C. Xu and J. L. Prince, “Snakes, shapes, and gradient vector flow,” IEEE Trans. Image Process., vol. 7, no. 3, pp. 359–369, Mar. 1998.
33. O. U. Aydin, “An evaluation of performance measures for arterial brain vessel segmentation,” BMC Med. Imag., vol. 21, no. 1, pp. 1–12, Dec. 2021.
34. A. A. Taha and A. Hanbury, “Metrics for evaluating 3D medical image segmentation: Analysis, selection, and tool,” BMC Med. Imag., vol. 15, no. 1, pp. 1–28, 2015.
35. O. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in Medical Image Computing and Computer-Assisted Intervention (Lecture Notes in Computer Science), vol. 9901, S. Ourselin, L. Joskowicz, M. Sabuncu, G. Unal, and W. Wells, Eds. Cham, Switzerland: Springer, 2016, doi: 10.1007/978-3-319-46723-8_49.

# 6.原文链接

👽[TaG-Net：Topology-Aware Graph Network for Centerline-Based Vessel Labeling](https://github.com/x-jeff/AI_Papers/blob/master/2024/TaG-Net：Topology-Aware%20Graph%20Network%20for%20Centerline-Based%20Vessel%20Labeling.pdf)

# 7.参考资料

1. [3D解剖丨头颈部、颅脑](https://www.sohu.com/a/553015665_121124565)