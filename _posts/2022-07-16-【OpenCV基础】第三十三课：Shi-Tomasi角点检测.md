---
layout:     post
title:      【OpenCV基础】第三十三课：Shi-Tomasi角点检测
subtitle:   cv::goodFeaturesToTrack
date:       2022-07-16
author:     x-jeff
header-img: blogimg/20220716.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Shi-Tomasi角点检测

Shi-Tomasi角点检测和[Harris角点检测](http://shichaoxin.com/2022/05/30/OpenCV基础-第三十二课-Harris角点检测/)的原理基本一模一样，唯一的不同在于响应函数R的计算，Shi-Tomasi角点检测的作者发现角点的稳定性其实和矩阵M的较小特征值有关，于是直接用较小的那个特征值就可以，这样就不用调整k值了：

$$R=\min (\lambda_1, \lambda_2)$$

# 2.`cv::goodFeaturesToTrack`

```c++
void goodFeaturesToTrack( 
	InputArray image, 
	OutputArray corners,
	int maxCorners, 
	double qualityLevel, 
	double minDistance,
	InputArray mask = noArray(), 
	int blockSize = 3,
	bool useHarrisDetector = false, 
	double k = 0.04 
);
```

参数详解：

1. `InputArray image`：输入图像，为8bit或32bit floating-point的单通道图像。
2. `OutputArray corners`：输出vector，保存检测到的角点。
3. `int maxCorners`：最多返回maxCorners个角点。如果检测到的角点数量大于maxCorners，则挑选前maxCorners个最优（strongest，个人理解就是按R值排序）的角点返回。如果maxCorners小于等于0，则没有返回数量限制，所有被检测到的角点都会被返回。
4. `double qualityLevel`：比如最优（即R值最大）的角点的R值为1500，如果qualityLevel=0.01，则R值小于$1500 \times 0.01=15$的角点都会被抛弃。
5. `double minDistance`：返回的任意两个角点之间最小的欧式距离。
6. `InputArray mask`：类型为`CV_8UC1`，大小和输入图像一样。mask用于限制只检测感兴趣的区域。
7. `int blockSize`：和[`cv::cornerHarris`](http://shichaoxin.com/2022/05/30/OpenCV基础-第三十二课-Harris角点检测/#3cvcornerharris)中的blockSize一样的含义。
8. `bool useHarrisDetector`：true则使用[Harris角点检测](http://shichaoxin.com/2022/05/30/OpenCV基础-第三十二课-Harris角点检测/)，false则使用Shi-Tomasi角点检测。
9. `double k`：和[`cv::cornerHarris`](http://shichaoxin.com/2022/05/30/OpenCV基础-第三十二课-Harris角点检测/#3cvcornerharris)中的k一样的含义。

# 3.代码地址

1. [Shi-Tomasi角点检测](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo33)