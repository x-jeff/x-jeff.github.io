---
layout:     post
title:      【OpenCV基础】第三十八课：Haar特征
subtitle:   Haar特征
date:       2023-04-26
author:     x-jeff
header-img: blogimg/20210609.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Haar特征

Haar特征（Haar-like feature）是用于物体识别的一种数字图像特征。它们因为与Haar小波转换（Haar wavelet）极为相似而得名，是第一种即时的人脸检测运算。

>Haar小波转换是由数学家Alfréd Haar于1909年提出的，是小波变换中最简单的一种变换，也是最早提出的小波变换。

历史上，直接使用图像的强度（就是图像每一个像素点的RGB值）使得特征的计算强度很大。Papageorgiou等人在1998年（Papageorgiou, Oren and Poggio, "A general framework for object detection", International Conference on Computer Vision, 1998.）提出可以使用基于Haar小波的特征而不是图像强度。Viola和Jones在2001年（Viola and Jones, "Rapid object detection using a boosted cascade of simple features", Computer Vision and Pattern Recognition, 2001.）进而提出了Haar特征。Haar特征使用检测窗口中指定位置的相邻矩形，计算每一个矩形的像素和并取其差值。然后用这些差值来对图像的子区域进行分类。

在Viola–Jones目标检测框架的检测阶段，一个与目标物体同样尺寸的检测窗口将在输入图像上滑动，在图像的每一个子区域都计算一个Haar特征。然后这个差值会与一个预先计算好的阈值进行比较，将目标和非目标区分开来。因为这样的一个Haar特征是一个弱分类器（它的检测正确率仅仅比随机猜测强一点点），为了达到一个可信的判断，就需要一大群这样的特征。在Viola-Jones目标检测框架中，就会将这些Haar特征组合成一个级联分类器，最终形成一个强分类群。

Haar特征最主要的优势是它的计算非常快速。使用[积分图](http://shichaoxin.com/2023/02/13/OpenCV基础-第三十七课-积分图计算/)，任意尺寸的Haar特征可以在常数时间内进行计算。

Viola和Jones提出的矩形Haar特征：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson38/38x1.png)

其中，(1)和(2)称为2矩形特征（2-rectangle feature），(3)称为3矩形特征（3-rectangle feature），(4)称为4矩形特征（4-rectangle feature）。

Lienhart和Maydt在2002年（Lienhart, R. and Maydt, J., "An extended set of Haar-like features for rapid object detection", ICIP02, pp. I: 900–903, 2002.）提出了倾斜45度的Haar特征：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson38/38x2.png)

这种对特征维度的扩充是为了提升对物体的检测。由于这些特征对一些物体的描述更为适合，这种扩充是有效的。例如，一个倾斜的特征可以描述一个倾斜45°的边缘。针对这种特征的计算，也提出了倾斜的积分图。

在计算Haar特征值时，用白色区域像素值的和减去黑色区域像素值的和，也就是说白色区域的权值为正值，黑色区域的权值为负值，而且权值与矩形区域的面积成反比，抵消两种矩形区域面积不等造成的影响。

Haar特征的取值受到特征模板的类别、位置以及大小这三种因素的影响，使得在一固定大小的图像窗口内，可以提取出大量的Haar特征。

# 2.参考资料

1. [哈尔特征（wiki百科）](https://zh.wikipedia.org/wiki/哈尔特征)
2. [图像特征提取之Haar特征](https://senitco.github.io/2017/06/15/image-feature-haar/)