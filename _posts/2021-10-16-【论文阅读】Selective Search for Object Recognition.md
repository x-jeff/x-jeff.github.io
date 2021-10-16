---
layout:     post
title:      【论文阅读】Selective Search for Object Recognition
subtitle:   Selective Search算法
date:       2021-10-16
author:     x-jeff
header-img: blogimg/20211016.jpg
catalog: true
tags:
    - Object Recognition
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Selective Search

本文只着重介绍论文中的第3部分，即Selective Search的实现细节。

>[R-CNN](http://shichaoxin.com/2021/09/20/论文阅读-Rich-feature-hierarchies-for-accurate-object-detection-and-semantic-segmentation/)使用Selective Search生成region proposals。

Selective Search需考虑以下设计因素：

**捕获所有尺寸（Capture All Scales）：**

图像中的目标可能是任意尺寸。并且，有些目标的边缘可能并不清晰。所以，Selective Search需要将所有尺寸的目标都纳入考虑。如Fig2所示，Selective Search找到了很多不同尺寸的目标。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/SelectiveSearch/1.png)

这部分通过一个分层算法来实现（an hierarchical algorithm）。

**多样性（Diversification）：**

区域的产生不是只有一个最优的策略。如Fig1所示，区域的产生可以来自很多不同的原因。在Fig1(b)中，我们可以通过颜色来区分这两只猫。在Fig1(c)中，我们可以通过纹理来区分叶子和变色龙。在Fig1(d)中，我们可以区分出车轮是因为它们是车的一部分，而不是因为颜色或者纹理相近。所以，在找寻这些目标时，有必要使用多样化的策略。并且，图像的本质是分层的（类似于PS中图层的概念），通俗点说就是多个目标位于不同的图层，之间可能会有遮挡。例如在Fig1(a)中，我们看不到完整的桌子、碗以及勺子等目标物体。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/SelectiveSearch/2.png)

此外，光线的明暗（例如阴影）以及颜色的亮度也会影响区域的产生。所以为了能够解决所有可能的情况，我们需要一个多样化的策略。

**快速计算（Fast to Compute）：**

Selective Search的目标是在实际的目标检测框架下，为其生成一系列可能的目标位置。所以Selective Search的速度不应该成为计算瓶颈。

## 1.1.Selective Search by Hierarchical Grouping

我们使用一个分层分组算法（a hierarchical grouping algorithm）作为Selective Search的基础。自下而上的分组（Bottom-up grouping）是一个非常流行的分割方法，所以我们将其应用于Selective Search。因为分组的过程本身就是分层（hierarchical），我们可以继续分组过程直至图像全部变为单一区域。这也刚好满足了捕获所有尺寸的要求。

因为区域包含的信息比像素更丰富，所以我们想尽可能的使用基于区域的特征。我们使用“Felzenszwalb and Huttenlocher (2004)”的方法来产生初始的划分区域。因为该方法速度快，且产生的区域不会横跨多个目标，很适合我们这种任务。

>“Felzenszwalb and Huttenlocher (2004)”的方法：Felzenszwalb, P. F.,&Huttenlocher, D. P. (2004). Efficient graph-based image segmentation. International Journal of Computer Vision, 59,167–181.

我们的分组过程见下。首先我们使用“Felzenszwalb and Huttenlocher (2004)”的方法来产生初始划分区域。然后我们应用贪心算法迭代的整合这些区域：首先计算所有相邻区域的相似度。相似度最高的一组相邻区域将会被整合在一起，即合并为一个新的区域。然后我们会重复这个步骤直至整幅图像变成一个单一的区域。详细的实现细节见下：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/SelectiveSearch/3.png)

>这里需要注意下，集合S里的元素随着区域的合并是不断减少的，直至其为空集。但是集合R里的元素是逐渐变多的，里面存储了最初始的划分区域以及后续每一次合并得到的新区域。这样的话就保证了R里面的区域有大有小，适用于不同尺寸的object。

为了加快计算，当把$r_i$和$r_j$合并为$r_t$时，我们希望可以根据$r_i$和$r_j$的特征便可计算出$r_t$的特征，而不需再访问图像像素。

## 1.2.多样性策略（Diversification Strategies）

多样性主要体现在三方面：

1. 使用不同的色彩空间。
2. 不同的相似度计算方式。
3. 改变初始划分区域。

接下来依次说明这三个方面。

👉第一个方面：**Complementary Colour Spaces.**

考虑到不同的场景和亮度。我们使用了8个色彩空间：

1. RGB色彩空间。
2. 灰度图像I。
3. [Lab色彩空间](https://baike.baidu.com/item/Lab%E9%A2%9C%E8%89%B2%E6%A8%A1%E5%9E%8B/3944053?fr=aladdin)。
4. rgI色彩空间。其中，r和g为RGB图像归一化后的r通道和g通道，再额外加上灰度图像I，凑成rgI三通道图像。
5. [HSV色彩空间](https://baike.baidu.com/item/HSV/547122?fr=aladdin)。
6. rgb色彩空间。即归一化后的RGB图像。
7. C色彩空间。详见论文：Geusebroek, J. M., van den Boomgaard, R., Smeulders, A. W. M., &
Geerts, H. (2001). Color invariance. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 23, 1338–1350.。
8. HSV色彩空间的H通道。

上述8种色彩空间的不变性见表1：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/SelectiveSearch/4.png)

其中，“+/-”表示部分不变性（“+”表示全部都有变化，“-”表示全部都没有变化）。分数，例如$\frac{1}{3}$表示三个通道中有一个通道具有不变性。表1是从亮度、阴影以及高亮三个维度来测试是否具有不变性（可以理解为：如果色彩空间变化后对这三个维度的敏感程度没有变化，即可视为不变）。从表1中可以看出，色彩空间从1到8的不变性是逐渐降低的。

算法每次执行的时候，只使用一种色彩空间。

👉第二个方面：**Complementary Similarity Measures.**

我们定义了四种互补的并且可以快速计算的相似性度量。相似度的值都被归一化为$[0,1]$，这也有助于不同相似度之间的组合。

$S_{colour}(r_i,r_j)$表示色彩相似度。计算色彩相似度，首先需要计算图像中每个区域的颜色直方图，例如区域$r_i$的颜色直方图为$C_i=\\{ c^1_i,…,c^n_i \\}$，分通道计算，每个通道的颜色直方图设置为25个bin。例如，对于三通道图像，共有$n=75$个bin。此外，对颜色直方图进行L1归一化处理（直方图相关知识请戳：[链接](http://shichaoxin.com/2021/08/02/OpenCV基础-第二十二课-直方图计算/)）。

$$S_{colour}(r_i,r_j)=\sum^n_{k=1} \min (c_i^k,c_j^k) \tag{1}$$

$r_i$和$r_j$合并后的区域$r_t$的颜色直方图可通过下式快速计算：

$$C_t =\frac{size(r_i) \times C_i + size(r_j) \times C_j}{size(r_i)+size(r_j)} \tag{2}$$

并且有：

$$size(r_t)=size(r_i)+size(r_j)$$

使用$S_{texture}(r_i,r_j)$表示纹理相似度。鉴于SIFT特征对不同材料识别的很好，因此使用类SIFT特征来度量纹理。作者计算了8个方向的高斯导数（高斯导数的讲解见本文第2部分）且有$\sigma = 1$。对于每个通道的每个高斯导数方向有计算其直方图，且设置$bin=10$。据此，我们便可得到区域$r_i$的纹理直方图$T_i = \\{ t_i^1,…,t_i^n \\}$。如果是三通道图像，则有$n=240$（$3 \times 10 \times 8=240$）。此外，对纹理直方图也进行了L1归一化处理。纹理相似度的计算见下：

$$S_{texture}(r_i,r_j)=\sum^n_{k=1} \min (t_i^k,t_j^k) \tag{3}$$

合并区域的纹理直方图的快速计算方式和式(2)类似，在此不再赘述。

大小相似度$S_{size}(r_i,r_j)$鼓励小区域更早合并。这样能使得S中的子区域都保持差不多的大小，不至于相差太多。这也可以防止一个区域一直在吞并其他区域，以至于其他区域没有吞并别人的机会。大小相似度的计算见下：

$$S_{size}(r_i,r_j)=1-\frac{size(r_i)+size(r_j)}{size(im)} \tag{4}$$

size(im)指的是图像的像素点个数。

$S_{fill}(r_i,r_j)$衡量区域$r_i$和区域$r_j$适合合并为一个区域的程度。我们用$BB_{ij}$表示可以包覆区域$r_i$和$r_j$的最小bounding box。$S_{fill}(r_i,r_j)$的计算见下：

$$fill(r_i,r_j)=1-\frac{size(BB_{ij})-size(r_i)-size(r_j)}{size(im)} \tag{5}$$

这里分母为size(im)是为了和式(4)保持一致。

最终相似度S的计算：

$$S(r_i,r_j)=a_1 S_{colour}(r_i,r_j)+a_2 S_{texture}(r_i,r_j) + a_3 S_{size}(r_i,r_j)+a_4 S_{fill}(r_i,r_j) \tag{6}$$

这里需要注意的是，$a_i$只有0和1两种取值，作者在此处并没有考虑使用加权的方式合并这些相似度。

👉第三个方面：**Complementary Starting Regions.**

多样性的第三个方面体现在初始划分区域的改变。据我们所知，“Felzenszwalb and
Huttenlocher (2004)”的方法是最快的，并且算法开源，划分的区域质量也比较高。作者在论文中说他找不到具有相似效率和性能的其他算法，所以其只使用了这一种划分初始区域的算法。但是不同的色彩空间会产生不同的初始划分区域。此外，作者也改变了“Felzenszwalb and
Huttenlocher (2004)”方法中的阈值参数k的值。

## 1.3.Combining Locations

有了集合R之后，我们需要从中挑选出哪些区域是可能包含object的。作者所用的方法是将最后一次合并得到的整幅图像的权重设置为1，前一次合并得到的区域权重设置为2，这样依次类推便可得到每个区域的权重，然后将每个区域的权重乘以一个[0,1]的随机数作为该区域的得分。最后按照得分的高低筛选出最后的location。具体筛选出得分前几的区域取绝于后续的算法设置，用户可自行确定。

这种方法使得一直在被合并的区域的权重会更大一些，算法认为如果一个区域一直在被合并，则这个区域很有可能是包含object的。

# 2.高斯导数

一维高斯函数方程：

$$G(x)=\frac{1}{\sqrt{2\pi} \sigma}e^{-\frac{x^2}{2\sigma^2}}$$

二维高斯函数方程：

$$G(x,y)=\frac{1}{2\pi \sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

二维高斯函数的一阶偏导数：

$$\frac{\partial G}{\partial x}=(-\frac{1}{2\pi \sigma^4})xe^{-\frac{x^2+y^2}{2\sigma^2}}$$

$$\frac{\partial G}{\partial y}=(-\frac{1}{2\pi \sigma^4})ye^{-\frac{x^2+y^2}{2\sigma^2}}$$

二维高斯函数的二阶偏导数：

$$\frac{\partial ^2 G}{\partial x^2}=(-\frac{1}{2\pi \sigma^4})(1-\frac{x^2}{\sigma^2})e^{-\frac{x^2+y^2}{2\sigma^2}}$$

$$\frac{\partial ^2 G}{\partial y^2}=(-\frac{1}{2\pi \sigma^4})(1-\frac{y^2}{\sigma^2})e^{-\frac{x^2+y^2}{2\sigma^2}}$$

$$\frac{\partial ^2 G}{\partial x \partial y}=(\frac{xy}{2\pi \sigma^6})e^{-\frac{x^2+y^2}{2\sigma^2}}$$

二维高斯函数的一阶、二阶梯度为：

$$\nabla G(x,y)=\lvert \frac{\partial G}{\partial x} \rvert + \lvert \frac{\partial G}{\partial y} \rvert$$

$$\nabla ^2 G(x,y)=\frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2}$$

二维高斯函数的一阶、二阶方向导数：

$$\frac{\partial G}{\partial \vec{l}}=\frac{\partial G}{\partial x} \cos \theta+\frac{\partial G}{\partial y}\sin \theta$$

$$\frac{\partial^2 G}{\partial \vec{l}^2}=\frac{\partial^2 G}{\partial x^2} \cos^2 \theta +\frac{\partial^2 G}{\partial y^2} \sin^2 \theta + 2 \frac{\partial^2 G}{\partial x \partial y}\cos \theta  \sin \theta$$

在selective search算法的纹理相似度计算部分，个人理解应该使用的是二维高斯函数的一阶方向导数，每个像素点与周边的八个相邻像素点构成了八个方向，即$\theta = \\{0°,45°,90°,135°,180°,225°,270°,315° \\}$。

# 3.参考资料

1. [高斯导数](https://jingyan.baidu.com/article/5bbb5a1bedf94413eba179ba.html)