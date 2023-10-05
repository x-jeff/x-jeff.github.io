---
layout:     post
title:      【OpenCV基础】第四十课：HOG特征检测
subtitle:   HOG，cv::HOGDescriptor，cv::HOGDescriptor::compute，HOG+SVM，cv::HOGDescriptor::getDefaultPeopleDetector，cv::HOGDescriptor::setSVMDetector，cv::HOGDescriptor::detectMultiScale
date:       2023-08-06
author:     x-jeff
header-img: blogimg/20220209.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.HOG特征检测

👉[【论文阅读】Histograms of Oriented Gradients for Human Detection](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/)。

# 2.`cv::HOGDescriptor`

使用`cv::HOGDescriptor`构建HOGDescriptor对象。

```c++
cv::HOGDescriptor::HOGDescriptor	(	
	Size 	_winSize,
	Size 	_blockSize,
	Size 	_blockStride,
	Size 	_cellSize,
	int 	_nbins,
	int 	_derivAperture = 1,
	double 	_winSigma = -1,
	int 	_histogramNormType = HOGDescriptor::L2Hys,
	double 	_L2HysThreshold = 0.2,
	bool 	_gammaCorrection = false,
	int 	_nlevels = HOGDescriptor::DEFAULT_NLEVELS,
	bool 	_signedGradient = false 
)	
```

参数详解：

1. `_winSize`：检测窗口大小。[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#65detector-window-and-context)和API的默认值都是$64 \times 128$。
2. `_blockSize`：block的大小（单位是像素）。[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/)和API默认值都是$(16,16)$。
3. `_blockStride`：block滑动步长（单位是像素）。[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/)和API默认值都是$(8,8)$。
4. `_cellSize`：cell的大小（单位是像素）。[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/)和API默认值都是$(8,8)$。
5. `_nbins`：直方图bin的数量。[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/)和API默认值都是9个。
6. `_derivAperture`：官方文档中没有对这个参数的详细解释。[`cv::Canny`](http://shichaoxin.com/2021/05/17/OpenCV基础-第十八课-Canny边缘检测算法/#2cvcanny)中有个参数`apertureSize`指的是[Sobel算子](http://shichaoxin.com/2021/03/01/OpenCV基础-第十六课-Sobel算子/)的size。这里该参数的默认值为1，即[Sobel算子](http://shichaoxin.com/2021/03/01/OpenCV基础-第十六课-Sobel算子/)的size为1，刚好就是[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#62gradient-computation)中认为最好的计算梯度的方式：$[-1,0,1]$。此外，该参数还被质疑并没有在HOGDescriptor class中被使用，详见github issue：[derivAperture, histogramNormType not used in HOGDescriptor](https://github.com/opencv/opencv/issues/9224)。
7. `_winSigma`：个人理解是[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#64normalization-and-descriptor-blocks)中Gaussian spatial window的$\sigma$值，用于对梯度幅值进行高斯加权，原文默认值是8。API中默认值是-1，猜测可能是不进行高斯加权的意思。在[这里](https://docs.opencv.org/2.4/modules/ocl/doc/feature_detection_and_description.html?highlight=hogdescriptor#ocl-hogdescriptor-hogdescriptor)找到的解释是“Gaussian smoothing window parameter.”，但[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#64normalization-and-descriptor-blocks)中提到过在梯度计算之前就对原始图像进行平滑会使性能下降。
8. `_histogramNormType`：归一化方式。[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#64normalization-and-descriptor-blocks)中提到了4种归一化方式：L2-norm、L2-Hys、L1-norm、L1-sqrt。这里默认是L2-Hys归一化。
9. `_L2HysThreshold`：L2-Hys归一化对最大值的限制，详见[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#64normalization-and-descriptor-blocks)，默认值为0.2。
10. `_gammaCorrection`：是否使用[gamma correction](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#61gammacolour-normalization)。
11. `_nlevels`：在[这里](https://docs.opencv.org/2.4/modules/ocl/doc/feature_detection_and_description.html?highlight=hogdescriptor#ocl-hogdescriptor-hogdescriptor)找到了对该参数的定义：“Maximum number of detection window increases.”，即检测窗口最多可增加的数量。
12. `_signedGradient`：是否使用带符号的梯度，详见[HOG原文](http://shichaoxin.com/2023/09/16/论文阅读-Histograms-of-Oriented-Gradients-for-Human-Detection/#63spatial--orientation-binning)。

# 3.`cv::HOGDescriptor::compute`

使用`compute`函数来计算HOG特征。

```c++
virtual void cv::HOGDescriptor::compute	(	
	InputArray 	img,
	std::vector< float > & 	descriptors,
	Size 	winStride = Size(),
	Size 	padding = Size(),
	const std::vector< Point > & 	locations = std::vector< Point >() 
)		const
```

参数详解：

1. `img`：用于计算HOG特征的输入图像，类型为`CV_8U`。
2. `descriptors`：计算得到的HOG特征向量，类型为`CV_32F`。如果检测窗口的大小为$64 \times 128$，block、cell、步长等参数都是默认值，那么一个窗口有$7 \times 15 = 105$个block，其特征向量长度为$105 \times 36 = 3780$。
3. `winStride`：检测窗口的步长，必须是block步长的倍数。
4. `padding`：用于对输入图像做padding。
5. `locations`：检测到的特征点。

核心实现代码：

```c++
HOGDescriptor detector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
vector<float> descriptors;
vector<Point> locations;
detector.compute(dst_gray, descriptors, Size(0, 0), Size(0, 0), locations);
```

# 4.HOG+SVM实现对行人检测

核心实现代码：

```c++
HOGDescriptor hog = HOGDescriptor();
hog.setSVMDetector(hog.getDefaultPeopleDetector());
vector<Rect> foundLocations;
hog.detectMultiScale(src, foundLocations, 1, Size(8, 8), Size(32, 32), 1.05, 2);
```

## 4.1.`cv::HOGDescriptor::getDefaultPeopleDetector`

```c++
static std::vector<float> cv::HOGDescriptor::getDefaultPeopleDetector	(		)	
```

返回为行人检测已经训练好的分类器的系数（适用于$64 \times 128$的检测窗口）。一个检测窗口得到的HOG特征向量长度为3780，但还要考虑到[偏置项$b$](http://shichaoxin.com/2021/01/03/机器学习基础-第十八课-支持向量机之核函数/)，所以这里这个函数返回的系数会是3781维的。

>opencv已经预训练好了一个线性SVM行人检测模型。

## 4.2.`cv::HOGDescriptor::setSVMDetector`

```c++
virtual void cv::HOGDescriptor::setSVMDetector	(	InputArray 	_svmdetector	)	
```

设置线性SVM分类器的系数。上述例子中我们直接使用opencv已经训练好的SVM模型。

## 4.3.`cv::HOGDescriptor::detectMultiScale`

用于检测输入图像中大小不同的目标。检测到的目标以矩形列表的形式返回。

```c++
virtual void cv::HOGDescriptor::detectMultiScale	(
	InputArray 	img,
	std::vector< Rect > & 	foundLocations,
	std::vector< double > & 	foundWeights,
	double 	hitThreshold = 0,
	Size 	winStride = Size(),
	Size 	padding = Size(),
	double 	scale = 1.05,
	double 	groupThreshold = 2.0,
	bool 	useMeanshiftGrouping = false 
)		const
```

1. `img`：输入图像。类型为`CV_8U`或`CV_8UC3`。
2. `foundLocations`：一个元素为rect的vector，每个rect对应一个被检测到的目标。
3. `foundWeights`：一个vector，对应每个被检测目标的confidence。
4. `hitThreshold`：是针对HOG特征向量到SVM分类平面的欧氏距离的一个阈值。通常设为0。当距离大于该阈值时，检测结果被接受，否则检测结果会被拒绝。设置此参数可有效降低假阳的出现。
5. `winStride`：检测窗口的步长，必须是block步长的倍数。
6. `padding`：对输入图像的padding操作。
7. `scale`：用于缩放检测窗口的大小。
8. `groupThreshold`：同一个目标可能会被多个检测框圈住，因此我们可以对这些检测框进行聚类。该参数用于指定聚类的半径。检测框中心距离小于`groupThreshold`的将会被聚类成一个框。如果该参数为0，则表示不进行聚类操作。opencv有的版本也将该参数写为`finalThreshold`。
9. `useMeanshiftGrouping`：聚类的算法。

另一种重载形式：

```c++
virtual void cv::HOGDescriptor::detectMultiScale	(
	InputArray 	img,
	std::vector< Rect > & 	foundLocations,
	double 	hitThreshold = 0,
	Size 	winStride = Size(),
	Size 	padding = Size(),
	double 	scale = 1.05,
	double 	groupThreshold = 2.0,
	bool 	useMeanshiftGrouping = false 
)		const
```

检测结果：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson40/40x1.png)

# 5.代码地址

1. [HOG特征检测](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo40)