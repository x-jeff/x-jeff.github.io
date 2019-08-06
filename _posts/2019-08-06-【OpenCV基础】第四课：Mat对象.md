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