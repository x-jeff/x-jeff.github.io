---
layout:     post
title:      【OpenCV基础】第四十五课：BRISK特征检测与匹配
subtitle:   cv::BRISK::create
date:       2024-09-17
author:     x-jeff
header-img: blogimg/20220212.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.BRISK

参见：[BRISK：Binary Robust invariant scalable keypoints](http://shichaoxin.com/2024/08/25/论文阅读-BRISK-Binary-Robust-invariant-scalable-keypoints/)。

# 2.OpenCV API

一共有3种重载形式。

```c++
static Ptr< BRISK > cv::BRISK::create	(	
    const std::vector< float > & radiusList,
    const std::vector< int > & numberList,
    float dMax = 5.85f,
    float dMin = 8.2f,
    const std::vector< int > & indexChange = std::vector< int >() 
)	
```

参数详解：

1. `radiusList`：是一个vector，用于定义围绕关键点（尺度为1）周围的一系列同心圆的半径。对于其他尺度的关键点，这些参数值会相应的自动调整。
2. `numberList`：是一个vector，分别对应在每个同心圆上的采样点数量（尺度为1的情况下）。对于其他尺度的情况，这些参数值会相应的自动调整。
3. `dMax`：短距离点对集合$\mathcal{S}$的阈值。
4. `dMin`：长距离点对集合$\mathcal{L}$的阈值。
5. `indexChange`：用于在描述符生成过程中重新排列比特的顺序。如果不提供该参数，则使用默认的索引顺序。这可以用于优化描述符的匹配效率。

```c++
static Ptr< BRISK > cv::BRISK::create	(
    int thresh,
    int octaves,
    const std::vector< float > & radiusList,
    const std::vector< int > & numberList,
    float dMax = 5.85f,
    float dMin = 8.2f,
    const std::vector< int > & indexChange = std::vector< int >() 
)	
```

参数详解：

1. `thresh`：AGAST检测阈值分数。AGAST是对[FAST](http://shichaoxin.com/2024/08/26/论文阅读-Machine-Learning-for-High-Speed-Corner-Detection/#21fast-features-from-accelerated-segment-test)的改进。
2. `octaves`：octave的数量。
3. `radiusList`：不再赘述。
4. `numberList`：不再赘述。
5. `dMax`：不再赘述。
6. `dMin`：不再赘述。
7. `indexChange`：不再赘述。

```c++
static Ptr< BRISK > cv::BRISK::create	(	
    int thresh = 30,
    int octaves = 3,
    float patternScale = 1.0f 
)	
```

参数详解：

1. `thresh`：不再赘述。
2. `octaves`：不再赘述。
3. `patternScale`：对默认的`radiusList`和`numberList`进行缩放。

可以从[opencv源码](https://github.com/opencv/opencv/blob/f503890c2b2ba73f4f94971c1845ead941143262/modules/features2d/src/brisk.cpp#L105)中看到是如何设置默认的`radiusList`和`numberList`，以及如何通过`patternScale`来进行缩放的：

```c++
virtual void setPatternScale(float _patternScale) CV_OVERRIDE
{
    patternScale = _patternScale;
    std::vector<float> rList;
    std::vector<int> nList;

    // this is the standard pattern found to be suitable also
    rList.resize(5);
    nList.resize(5);
    const double f = 0.85 * patternScale;

    rList[0] = (float)(f * 0.);
    rList[1] = (float)(f * 2.9);
    rList[2] = (float)(f * 4.9);
    rList[3] = (float)(f * 7.4);
    rList[4] = (float)(f * 10.8);

    nList[0] = 1;
    nList[1] = 10;
    nList[2] = 14;
    nList[3] = 15;
    nList[4] = 20;

    generateKernel(rList, nList, (float)(5.85 * patternScale), (float)(8.2 * patternScale));
}
```

# 3.代码地址

1. [BRISK特征检测与匹配](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo45)