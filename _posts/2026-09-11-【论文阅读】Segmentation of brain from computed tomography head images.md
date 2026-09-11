---
layout:     post
title:      【论文阅读】Segmentation of brain from computed tomography head images
subtitle:   brain segmentation
date:       2026-09-11
author:     x-jeff
header-img: blogimg/20200828.jpg
catalog: true
tags:
    - Medical Imaging
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.INTRODUCTION

本文提出了一种在CT数据中提取脑组织的方法。

# 2.METHOD

脑组织主要由灰质（GM）和白质（WM）构成。在脑部CT图像中，骨组织具有最高的CT值，其次依次为灰质、白质、脑脊液（CSF）和空气。某些非脑组织，例如鼻窦区域以及肌肉，其CT值可能与灰质或白质相近。CT颅脑扫描通常层厚较大（>=5mm），这导致相邻axial图像之间具有一定的空间相关性。但是，与层厚较小的MRI数据不同，不能简单假设整个脑组织一定会形成图像中的最大连通域。

对于一个三维体数据，x方向为从受检者的左侧指向右侧，y方向为从前方到后方，z方向为从上方到下方。位置$(x,y,z)$处的体素的CT值记为$g(x,y,z)$。

算法的整体流程见下图：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/BrainSeg/1.png)

## 2.1.Choose a reference image $g(x,y,z_0)$

参考图像为一张axial图像，且满足如下要求：

* 图像中同时包含白质、灰质、脑脊液、空气以及颅骨组织。
* 能够通过解剖学特征较容易的从整个CT体数据中提取出来。
* 在选定的参考图像中，颅骨内部的灰质和白质所占面积比例，在不同受检者之间应该比较稳定，差异较小。

在实际应用中，可以用一张包含第三脑室、但不包含眼球的axial切片来近似这一参考图像，如Fig2(a)所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/BrainSeg/2.png)

参考图像的层数记为$z_0$。

>个人注解：作者并没有在论文中描述如何自动确定参考图像。

## 2.2.Determine region of interest

参考图像中的ROI定义为由颅骨包围的内部区域，后续记为head mask。ROI的确定方法如下：

1. 确定一个阈值用于二值化参考图像。使用FCM（Fuzzy C-Means）聚类将参考图像的CT值分成4类，其中第1类具有最低的CT值。将第1类中的最大CT值加5，记为backG。
2. 根据backG阈值对参考图像进行二值化，得到初始head mask。CT值小于backG的体素为背景，否则为前景。
3. 寻找最大的前景连通域，并将其他所有前景连通域都设置为背景。
4. 填充最大前景连通域内的孔洞，得到最终的ROI，如Fig2(b)所示。

>个人注解：作者并没有在论文中详细说明FCM聚类的超参数设置。

## 2.3.Calculate low and high thresholds

低阈值的作用是从脑区域中排除空气和脑脊液，其计算步骤如下：

1. 在参考图像的head mask内，根据CT值直方图，将CT值分为4类（同样使用FCM聚类），分别对应空气/脑脊液、白质、灰质、骨组织。将第4类，也就是骨组织中最小的CT值记为minBone。
2. 低阈值的计算方式为：$lowThresh = meanC_1 + \alpha_1 * sdC_1$。其中，$meanC_1$和$sdC_1$分别表示第1类的平均值和标准差，$\alpha_1$是一个取值范围为0~3的常数。当希望尽量减少脑组织被误分类成非脑组织时，$\alpha_1$应取较小的值，例如小于1；反之，如果更关注把脑组织和非脑组织分得更彻底，则$\alpha_1$应取较大的值，例如大于2。

高阈值用于排除那些比灰质和白质更亮的骨组织。高阈值的确定步骤如下：

1. 在参考图像的head mask内，寻找满足以下条件的像素对：
    * 这两个像素是8邻域相连的。
    * 其中一个像素的CT值大于等于minBone，即认为它属于骨组织。
    * 另一个像素的CT值小于minBore，但大于lowThresh，即认为它属于白质或灰质。
2. 对于第1步找到的所有符合要求的像素对，计算所有CT值大于等于minBore的像素的平均CT值，记为brightAvg。类似地，计算所有CT值小于minBore的像素的平均CT值，记为darkAvg。
3. 高阈值的计算为：$highThresh = \alpha * brightAvg + ( 1-\alpha ) * darkAvg$。其中，$\alpha$是一个权重参数，取值范围为0到1。如果误删脑组织的代价高于误把非脑组织包含进来的代价，那么$\alpha$应该取大于0.5的值。如果两类错误的代价同等重要，或者希望使总体分类误差最小，则设$\alpha=0.5$。

## 2.4.Perform binarization

对整个三维体数据做二值化：

$$binM(x,y,z) = \begin{cases} 1, & lowThresh \leqslant g(x,y,z) \leqslant highThresh \\ 0, & otherwise \end{cases}$$

Fig2(a)依据上式进行二值化的结果见如下Fig3(a)：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/BrainSeg/3.png)

## 2.5.Find brain candidates

对于每一张axial图像，都按照第2.2部分描述的方法生成各自的head mask。在第2.4部分，我们得到每一张axial图像的所有前景连通域，针对每个前景连通域，计算其到该axial图像的head mask的背景像素的最小距离，如果最小距离大于某个阈值（比如10mm），则认为该前景连通域是一个脑候选区域，否则，将其设置为背景，即认为它属于颅骨等非脑组织。Fig3(b)展示了参考图像的脑候选区域，Fig4展示了参考图像下方的一张axial图像的脑候选区域。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/BrainSeg/4.png)

注意，在Fig4中，仍有一些非脑区域没有被去除，这些区域将在下一阶段继续处理。

## 2.6.Propagate brain masks

依次检查每一张层数$z > z_0$的axial图像的脑候选区域，比如对于层数为$z_0 + 1$的axial图像，具体方法如下：

1. 对于层数为$z_0 + 1$的axial图像中的某一前景连通域，假设其包含$N$个前景像素，对于这个前景连通域中的每个点$(x_i,y_i)$，查看其在第$z_0$层是否为前景，将两层都为前景的像素数量统计为$N_1$。
2. 如果$N_1 < \beta N$，说明第$z_0+1$层的这个连通域与第$z_0$层差异过大，因此将该前景连通域纠正为背景。$\beta$的取值范围为0到1，通常取0.5。

>个人注解：作者并没有在论文中提及，对于$z < z_0$的层，是否也要做同样的处理。

Fig4在经过第2.6部分处理后，得到Fig5：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/BrainSeg/5.png)

# 3.RESULTS

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/BrainSeg/6.png)

# 4.DISCUSSION AND CONCLUSION

不再详述。

# 5.原文链接

👽[Segmentation of brain from computed tomography head images](https://github.com/x-jeff/AI_Papers/blob/master/2026/Segmentation%20of%20brain%20from%20computed%20tomography%20head%20images.pdf)