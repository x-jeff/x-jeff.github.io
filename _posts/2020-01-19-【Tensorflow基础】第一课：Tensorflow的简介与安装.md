---
layout:     post
title:      【Tensorflow基础】第一课：Tensorflow的简介与安装
subtitle:   Tensorflow的简介，安装Tensorflow
date:       2020-01-19
author:     x-jeff
header-img: blogimg/20200119.jpg
catalog: true
tags:
    - Tensorflow Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.TensorFlow简介

**TensorFlow**是由谷歌开发的一款开源的深度学习框架。其底层代码为C++，提供CPU和GPU两种版本。并且可以通过**TensorBoard**可视化网络结构和参数。

# 2.安装tensorflow

本机环境为`Mac OS`。为了避免环境之间互相污染，方便管理，我们新建一个conda虚拟环境用于学习tensorflow：

1. 新建一个conda虚拟环境：`conda create -n tensorflow python=3.6`
2. 进入虚拟环境：`source activate tensorflow`
3. 安装tensorflow（默认安装最新版本）：
	* CPU版本：`pip install tensorflow`
	* GPU版本：`pip install tensorflow-gpu`
4. （可选）安装特定版本的tensorflow（自动卸载之前已安装的版本）：
	* CPU版本：`pip install tensorflow==<版本号>`
	* GPU版本：`pip install tensorflow-gpu==<版本号>`

>👉[常见conda命令](http://shichaoxin.com/2019/12/26/conda-常用的conda命令/)

## 2.1.更新tensorflow版本

1. 卸载旧版本：`pip uninstall tensorflow`
2. 默认安装最新版本：`pip install tensorflow`