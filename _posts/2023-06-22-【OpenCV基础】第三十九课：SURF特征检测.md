---
layout:     post
title:      【OpenCV基础】第三十九课：SURF特征检测
subtitle:   SURF，cv::xfeatures2d::SURF::create
date:       2023-06-22
author:     x-jeff
header-img: blogimg/20181104.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.SURF特征检测

SURF讲解：[【论文阅读】SURF：Speeded Up Robust Features](http://shichaoxin.com/2023/08/18/论文阅读-SURF-Speeded-Up-Robust-Features/)。

# 2.API

```c++
static Ptr<SURF> cv::xfeatures2d::SURF::create(
	double hessianThreshold = 100,
	int nOctaves = 4,
	int nOctaveLayers = 3,
	bool extended = false,
	bool upright = false 
)	
```

参数详解：

1. `hessianThreshold`：像素点的响应值（即近似的Hessian矩阵行列式）超过这个阈值才被认为是兴趣点。通常设置在300～500之间。
2. `nOctaves`：octave的数量。
3. `nOctaveLayers`：每个octave内的层数。
4. `extended`：如果为true，则使用SURF-128；如果为false，则使用常规的SURF（即SURF-64）。
5. `upright`：如果为true，则使用U-SURF（不具有旋转不变性，计算速度更快）；如果是false，则使用常规的SURF（具有旋转不变性）。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson39/39x1.png)

# 3.代码地址

1. [SURF特征检测](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo39)