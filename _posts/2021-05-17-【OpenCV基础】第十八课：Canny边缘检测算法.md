---
layout:     post
title:      【OpenCV基础】第十八课：Canny边缘检测算法
subtitle:   Canny边缘检测算法，cv::Canny
date:       2021-05-17
author:     x-jeff
header-img: blogimg/20210517.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Canny边缘检测算法

Canny边缘检测算法是1986年提出的。其主要分为5步：

1. [高斯模糊](http://shichaoxin.com/2020/03/03/OpenCV基础-第九课-图像模糊/#3高斯模糊)。
2. [灰度转换](http://shichaoxin.com/2019/04/01/OpenCV基础-第二课-加载-修改-保存图像/#4修改图像)。
3. 计算梯度（通常使用[Sobel算子](http://shichaoxin.com/2021/03/01/OpenCV基础-第十六课-Sobel算子/)）。
4. 非最大信号抑制。
5. 高低阈值输出二值图像。

前三步在之前的博客中都有所介绍，现在着重讲解下后两步。

👉第4步：非最大信号抑制：

经过前三步的处理之后，我们得到的边缘可能很模糊，而这一步就是起到将边缘“瘦身”的作用。其主要做法为：

1. 将当前像素的梯度强度（即灰度值）与沿正负梯度方向上的两个像素进行比较。
2. 如果当前像素的梯度强度与另外两个像素相比最大，则该像素点保留为边缘点，否则该像素点将被抑制（即灰度值置为0）。

![](https://github.com/x-jeff/BlogImage/raw/master/OpenCVSeries/Lesson18/18x1.png)

如上图所示，共有9个相邻的像素点（8个黄色点+1个红色点）。以中心像素点（红色点）为例，如果梯度方向刚好是$45^o$的整倍数，那么中心像素点刚好在梯度方向上都有相邻的两个像素。但如果梯度方向不是$45^o$的整倍数，则其会横穿两个像素之间，因此我们需要对其进行线性插值来精确计算。以上图为例，假设蓝色箭头所示方向为梯度方向，则通过线性插值得到的两个像素点（蓝色点）的计算方式为：

$$g_{up}(i,j)=(1-t)\cdot g_{xy}(i,j+1)+t \cdot g_{xy}(i-1,j+1)$$

$$g_{down}(i,j)=(1-t)\cdot g_{xy}(i,j-1)+t \cdot g_{xy}(i+1,j-1)$$

如果：

$$g_{xy}(i,j) \geqslant g_{up}(i,j) \  and \  g_{xy}(i,j) \geqslant g_{down}(i,j)$$

则$g_{xy}(i,j)$被认为是边缘，否则应该被抑制，即灰度值置为0。

👉第5步：高低阈值输出二值图像：

定义一个高阈值和一个低阈值，大于高阈值的像素点都被检测为边缘，而低于低阈值的像素点都被检测为非边缘。对于像素值位于高低阈值之间的像素点，如果与确定为边缘的像素点邻接，则判定为边缘，否则为非边缘。

推荐的高低阈值比值为2:1或3:1。

# 2.`cv::Canny`

```c++
void Canny( 
	InputArray image, 
	OutputArray edges,
	double threshold1, 
	double threshold2,
	int apertureSize = 3, 
	bool L2gradient = false 
);
```

参数详解：

1. `InputArray image`：8-bit的输入图像，例如灰度图。
2. `OutputArray edges`：输出的边缘检测图像。一般为二值图像，背景是黑色。
3. `double threshold1`：低阈值，常取高阈值的$\frac{1}{2}$或者$\frac{1}{3}$。
4. `double threshold2`：高阈值。
5. `int apertureSize`：[Sobel算子](http://shichaoxin.com/2021/03/01/OpenCV基础-第十六课-Sobel算子/)的size，通常取3。
6. `bool L2gradient`：计算梯度值的方式。`true`表示使用[L2正则化](http://shichaoxin.com/2021/03/01/OpenCV基础-第十六课-Sobel算子/#2sobel算子)（更精确）；`false`表示使用[L1正则化](http://shichaoxin.com/2021/03/01/OpenCV基础-第十六课-Sobel算子/#2sobel算子)（计算量更小）。

# 3.代码地址

1. [Canny边缘检测算法](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo18)

# 4.参考资料

1. [Canny边缘检测算法](https://zhuanlan.zhihu.com/p/99959996)
2. [Canny边缘检测算法解析](https://blog.csdn.net/qq_29462849/article/details/81050212)