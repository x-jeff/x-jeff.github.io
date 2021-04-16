---
layout:     post
title:      【OpenCV基础】第十七课：Laplace算子
subtitle:   Laplace算子，cv::Laplacian，cv::convertScaleAbs
date:       2021-04-16
author:     x-jeff
header-img: blogimg/20210416.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Laplance算子

[Sobel算子](http://shichaoxin.com/2021/03/01/OpenCV基础-第十六课-Sobel算子/)属于一阶微分算子，利用了一阶导数，图像在边缘处的一阶导数值最大。而**Laplace算子**属于二阶微分算子，利用了二阶导数，图像在边缘处的二阶导数为零：

![](https://github.com/x-jeff/BlogImage/raw/master/OpenCVSeries/Lesson17/17x1.png)

离散函数的导数退化成了差分，一维一阶差分公式和二阶差分公式分别为：

$$\frac{\partial f }{\partial x}=f'(x)=f(x+1)-f(x)$$

$$\frac{\partial ^2 f }{\partial x^2}=f''(x)=f'(x)-f'(x-1)=f(x+1)+f(x-1)-2f(x)$$

上述是一维情况下，那么在二维函数$f(x,y)$中，$x,y$两个方向的二阶差分分别为：

$$\frac{\partial ^2 f}{\partial x^2}=f(x+1,y)+f(x-1,y)-2f(x,y)$$

$$\frac{\partial ^2 f}{\partial y^2}=f(x,y+1)+f(x,y-1)-2f(x,y)$$

所以Laplace算子的差分形式为：

$$\nabla ^2 f(x,y)=f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)$$

写成filter的形式：

$$\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \\ \end{bmatrix}$$

# 2.API

```c++
void Laplacian( 
	InputArray src, 
	OutputArray dst, 
	int ddepth,
	int ksize = 1, 
	double scale = 1, 
	double delta = 0,
	int borderType = BORDER_DEFAULT 
	);
```

参数解释：

1. `InputArray src`：输入图片。
2. `OutputArray dst`：输出图片。
3. `int ddepth`：输出图片的[位图深度](http://shichaoxin.com/2019/06/02/OpenCV基础-第三课-掩膜操作/#331位图深度)。
4. `int ksize`：filter大小，必须为正奇数。
5. `double scale`：filter中的每一个值乘以scale。
6. `double delta`：filter中的每一个值加上delta。
7. `int borderType`：[边界处理方式](http://shichaoxin.com/2020/12/11/OpenCV基础-第十五课-边缘处理/)。

# 3.图像处理步骤

1. 高斯模糊（去噪声）：`GaussianBlur()`。
2. 转化为灰度图像：`cvtColor()`。
3. 应用Laplace算子：`Laplacian()`。
4. 图像取绝对值：`convertScaleAbs()`。
5. 显示结果。

下左为原图，下右为应用Laplace算子的效果：

![](https://github.com/x-jeff/BlogImage/raw/master/OpenCVSeries/Lesson17/17x2.png)

## 3.1.`convertScaleAbs`

```c++
void convertScaleAbs(
	InputArray src, 
	OutputArray dst,
	double alpha = 1, 
	double beta = 0
	);
```

`cv::convertScaleAbs()`对整个图像数组中的每一个元素进行如下操作：

$$dst_i=saturate_{uchar}( \mid \alpha * src_i + \beta \mid)$$

# 4.代码地址

1. [Laplace算子](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo17)

# 5.参考资料

1. [opencv学习 边缘检测 --拉普拉斯算子（Laplace）](https://www.huaweicloud.com/articles/9e6c73bff404a6f0acdef3cfea9b4f51.html)