---
layout:     post
title:      【OpenCV基础】第一课：OpenCV环境配置
subtitle:   环境配置
date:       2019-01-19
author:     x-jeff
header-img: blogimg/20190119.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.OpenCV简介
OpenCV是一个计算机视觉的开源库。英文全称是：Open Source Computer Vision Library。

常用的OpenCV的核心模块：

1. Image Process
2. Camera Calibration and 3D Reconstruction
3. Video Analysis
4. Object Detection
5. Machine Learning
6. Deep Learning
7. GPU Acceleration
8. ......

官方网站：[https://opencv.org](https://opencv.org)  
GitHub：[https://github.com/opencv](https://github.com/opencv)

>相关知识补充：
>
>**SDK：**软件开发工具包（Software Development Kit）一般都是一些被软件工程师用于为特定的软件包、软件框架、硬件平台、操作系统等建立应用软件的开发工具的集合。（资料来源：[百度百科：SDK](https://baike.baidu.com/item/sdk/7815680)）
>
>**GPU：**图形处理器（Graphics Processing Unit，GPU），又称显示核心、视觉处理器、显示芯片，是一种专门在个人电脑、工作站、游戏机和一些移动设备（如平板电脑、智能手机等）上图像运算工作的微处理器。显卡的处理器称为图形处理器（GPU），它是显卡的“心脏”，与CPU类似，只不过GPU是专为执行复杂的数学和几何计算而设计的。GPU已经不再局限于3D图形处理了，GPU通用计算技术发展已经引起业界不少的关注，事实也证明在浮点运算、并行计算等部分计算方面，GPU可以提供数十倍乃至于上百倍于CPU的性能。GPU相当于专用于图像处理的CPU，正因为它专，所以它强，在处理图像时它的工作效率远高于CPU，但是CPU是通用的数据处理器，在处理数值计算时是它的强项，它能完成的任务是GPU无法代替的，所以不能用GPU来代替CPU。（资料来源：[百度百科：GPU](https://baike.baidu.com/item/图形处理器?fromtitle=gpu&fromid=105524)）
>
>**GPU通用计算方面的标准：**目前有OpenCL、CUDA、ATI STREAM。



# 2.OpenCV配置

电脑环境：**Mac+Clion+C++**。

## 2.1.OpenCV下载

OpenCV的两种下载方法：

1. 官网下载（官网地址见上文）。
2. 通过HomeBrew下载。

我个人是通过HomeBrew下载的，也推荐大家使用这种下载方式，方便包的管理和更新。具体流程见下：

### 2.1.1.安装HomeBrew
>HomeBrew中文官方网址：[HomeBrew](https://brew.sh/index_zh-cn.html)

打开终端，输入以下命令：

`
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
`

### 2.1.2.下载CMake

CMake官方网站：[CMake](https://cmake.org)

点击`Download`进入下载页面，选择适合的`Platform`进行下载，我选择的是`Mac OS`：
![](https://github.com/x-jeff/BlogImage/raw/master/OpenCVSeries/Lesson1/1x1.jpg)

但是，下载完成之后，在终端依旧无法使用cmake命令，这时需要进一步的设置：

![](https://github.com/x-jeff/BlogImage/raw/master/OpenCVSeries/Lesson1/1x2.jpg)

点击`How to Install For Command Line Use`，出现以下提示：

![](https://github.com/x-jeff/BlogImage/raw/master/OpenCVSeries/Lesson1/1x3.jpg)

选择第二种方法，在终端输入：

`
sudo "/Applications/CMake.app/Contents/bin/cmake-gui" --install
`

这样使得每次打开终端，都可以正确识别cmake命令。

当然，也可以选择使用HomeBrew下载CMake，在终端输入如下命令即可：

`
brew install cmake
`

### 2.1.3.下载OpenCV

使用HomeBrew下载OpenCV，在终端输入以下命令：

`
brew install opencv
`

下载的为最新版本的OpenCV，我下载的版本为4.0.1。

## 2.2.OpenCV在Clion中的配置

打开Clion，新建一个C++项目。

修改`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.13) ##cmake版本
project(CDemo2) ##项目名称
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)  ##main.cpp改为自己定义的名字
add_executable(CDemo2 ${SOURCE_FILES}) ##CDemo2改为自己的项目名称
target_link_libraries(CDemo2 ${OpenCV_LIBS})  ##CDemo2改为自己的项目名称
```

## 2.3.代码测试

```c++
#include <iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main() {
    Mat p1=imread("/Users/xinshichao/PersonalWork/C++Demo/Pictures/p1.jpeg"); //改成自己的图片路径，尽量使用绝对路径
    if(!p1.data){
        printf("could not find the image...\n");
    }
    namedWindow("output",WINDOW_AUTOSIZE);
    imshow("output",p1);
    waitKey(0);
    return 0;
}
```

输出结果：
![](https://github.com/x-jeff/BlogImage/raw/master/OpenCVSeries/Lesson1/1x4.jpg)

OpenCV配置成功！

# 3.代码地址

1. [OpenCV测试代码](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo1)
