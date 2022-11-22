---
layout:     post
title:      【OpenCV基础】第四课：Mat对象
subtitle:   Mat对象，复制
date:       2019-08-06
author:     x-jeff
header-img: blogimg/20190806.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Mat对象和IplIamge对象

* Mat对象是OpenCV2.0之后引进的图像数据结构、自动分配内存、不存在内存泄漏的问题，是面向对象的数据结构。分为两个部分：**头部**和**数据部分**。
* IplIamge是从2001年OpenCV发布之后就一直存在，是C语言风格的数据结构，需要开发者自己分配与管理内存、对大的程序使用它容易导致内存泄漏问题。

# 2.Mat对象构造函数

Mat对象有以下6种构建方法：

1. `Mat(int rows,int cols,int type)`(常用)
2. `Mat(Size size,int type)`(常用)
3. `Mat(int rows,int cols,int type,const Scalar &s)`
4. `Mat(Size size,int type,const Scalar &s)`
5. `Mat(int ndims,const int* sizes,int type)`
6. `Mat(int ndims,const int* sizes,int type,const Scalar &s)`

## 2.1.图像的其他构造方法

**方法一：**

`img.create(size,type)`，但是`.create`不能赋值，赋值需要用`img=Scalar(0,0,0)`。

**方法二：**

`Mat::eye(3,3,type)`构造一个3行3列（即3*3）的图像，并且对角线上每个像素点的第一个通道赋值为1，其余均为0，对如上语句，如果type为3通道图像的话，得到的图像数组矩阵见下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson4/4x1.png)

类似的还有：

* `Mat::ones()`：每个像素的第一个通道为1，其余通道为0。
* `Mat::zeros()`：每个像素的每个通道都为0。

# 3.部分复制和完全复制

👉**部分复制：**一般情况下只会复制Mat对象的头和指针部分，不会复制数据部分。

👉**完全复制：**把Mat对象的头部和数据部分一起复制。

```c++
Mat A=imread(imgFilePath);
Mat B(A);//部分复制
Mat F=A.clone();//完全复制
Mat G;
A.copyTo(G);//完全复制
```

(部分复制：共用一个矩阵，即一张图，该图如果发生改变，A和B都将改变。而完全复制是各自用各自的数据部分，互不干扰。)

## 3.1.`copyTo()`

`copyTo()`有三种声明方式：

```c++
//1
Mat clone() const CV_NODISCARD;
//2
void copyTo( OutputArray m ) const;
//3
void copyTo( OutputArray m, InputArray mask ) const;
```

其中第三种方式还可用于图像融合，mask的尺寸必须和原图像相同，它的非零元素表示需要复制的矩阵元素，mask必须是`CV_8U`类型的，可以有1个或多个通道。例如：

```c++
src.copyTo(dst,mask);
```

如果mask在像素点$(i,j)$的像素值不为0，则把src在像素点$(i,j)$的像素值直接赋给dst的$(i,j)$处；如果mask在像素点$(i,j)$的像素值为0，则dst保留其在$(i,j)$处的像素值。

# 4.代码地址

1. [Mat对象](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo4)
2. [OpenCV中mat::copyto( )函数使用方法](https://blog.csdn.net/NNNNNNNNNNNNY/article/details/52296827)