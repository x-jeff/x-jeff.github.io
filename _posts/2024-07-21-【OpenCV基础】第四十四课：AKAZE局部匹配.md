---
layout:     post
title:      【OpenCV基础】第四十四课：AKAZE局部匹配
subtitle:   非线性尺度空间，KAZE，AKAZE
date:       2024-07-21
author:     x-jeff
header-img: blogimg/20210301.jpg
catalog: true
tags:
    - OpenCV Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.KAZE

KAZE特征算法出自论文“KAZE Features”，其在2012年的ECCV会议中由一名法国学者提出。KAZE这一名字是为了纪念尺度空间分析之父Iijima（日本学者），KAZE在日语中是“风”的意思，这是因为风是空气在空间中的非线性流动过程。

传统的图像特征检测和描述算法，如[SIFT](http://shichaoxin.com/2022/12/29/OpenCV基础-第三十六课-SIFT特征检测/)和[SURF](http://shichaoxin.com/2023/06/22/OpenCV基础-第三十九课-SURF特征检测/)，使用高斯平滑来构建线性尺度空间。然而，线性尺度空间在处理高频细节时可能会丢失一些重要信息。KAZE通过使用非线性扩散滤波器来构建非线性尺度空间，能够更好地保持图像细节和边缘信息。

## 1.1.非线性扩散滤波（Nonlinear Diffusion Filtering）

非线性扩散以一种非线性的方式对图像进行平滑，而我们常见的高斯平滑则是以一种线性的方式对图像进行平滑。非线性扩散方法通常由非线性偏微分方程（partial differential equations，PDEs）描述。经典的非线性扩散公式见下：

$$\frac{\partial L}{\partial t} = div (c(x,y,t)\cdot \nabla L) \tag{1}$$

$div$是散度，在介绍散度之前，我们先了解下什么是向量场。向量场将空间中的每个点与一个向量关联起来，在数学上，向量场是一个函数，输入是空间中的一个点，输出是一个向量。比如某个区域内的风速或水流速度，在每个点上，向量的方向表示流体的流动方向，向量的大小表示流速。假设在三维空间中，点用坐标$(x,y,z)$表示，那么向量场$\mathbf{F}$可表示为：

$$\mathbf{F}(x,y,z) = \left( \mathbf{F}_x(x,y,z), \mathbf{F}_y(x,y,z), \mathbf{F}_z(x,y,z) \right) \tag{2}$$

其中，$(\mathbf{F}_x,\mathbf{F}_y,\mathbf{F}_z)$是描述向量场在点$(x,y,z)$的三个分量。那么其散度可定义为：

$$div \  \mathbf{F} = \nabla \cdot \mathbf{F} =  \frac{\partial \mathbf{F}_x}{\partial x} + \frac{\partial \mathbf{F}_y}{\partial y} + \frac{\partial \mathbf{F}_z}{\partial z} \tag{3}$$

举个例子，我们的向量场可以是如下形式：

$$\mathbf{F}(x,y,z) = (x^2y,yz,z^2x) \tag{4}$$

其散度为：

$$\begin{align} div \  \mathbf{F} &= \frac{\partial \mathbf{F}_x}{\partial x} + \frac{\partial \mathbf{F}_y}{\partial y} + \frac{\partial \mathbf{F}_z}{\partial z} \\&= \frac{\partial (x^2y)}{\partial x} + \frac{\partial (yz)}{\partial y} + \frac{\partial (z^2x)}{\partial z} \\&= 2xy + z+2zx \end{align} \tag{5}$$

回到式(1)，$L$是图像亮度（像素值），$\nabla L$是图像梯度。$c$是传导函数（conductivity function）。时间$t$是一个尺度参数，用于控制图像平滑的程度和进展。$t$越大，图像表示形式就越简单。

Perona和Malik提出使函数$c$依赖于梯度幅度，以减少在边缘位置的扩散，从而鼓励在区域内平滑而不是跨边界平滑。因此，函数$c$被定义为（P-M方程）：

$$c(x,y,t) = g(\lvert \nabla L_{\sigma} (x,y,t) \rvert) \tag{6}$$

其中，$L_{\sigma}$是经过高斯平滑后的图像$L$。传导函数$g$一共有3种常见的形式（前两种由Perona和Malik提出，最后一种由Weickert提出）：

$$g_1 = \exp \left( - \frac{\lvert \nabla L_{\sigma} \rvert ^2}{k^2} \right) \tag{7}$$

$$g_2 = \frac{1}{1+\frac{\lvert \nabla L_{\sigma} \rvert^2}{k^2}} \tag{8}$$

$$\begin{equation}
g_3 =
\begin{cases}
1 & , \lvert \nabla L_\sigma \rvert ^2 = 0 \\ 
1 - \exp \left( - \frac{3.315}{(\lvert \nabla L_\sigma\rvert / k)^8} \right) & , \lvert \nabla L_\sigma \rvert^2 > 0 
\end{cases}
\end{equation} \tag{9}$$

其中，$k$是对比度因子（contrast factor）。$k$是一个超参数，根据经验，通常设置为平滑后图像梯度直方图70%百分位上的值。如下图所示，以$g_1$为例，$k$值越大，只有较大的梯度才被考虑：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson44/44x1.png)

接下来，我们希望可以对式(1)进行求解，以得到在不同时间$t$下的图像$L$。但式(1)这样的非线性偏微分方程并没有[解析解](http://shichaoxin.com/2019/06/30/机器学习基础-第六课-线性回归/#22参数估计)，因此我们通过AOS（Additive Operator Splitting）方法来近似求解。我们将式(1)离散化表示为：

$$\frac{L^{i+1}-L^i}{\tau} = \sum_{l=1}^m A_l (L^i) L^{i+1} \tag{10}$$

其中，$A_l$是图像每个维度的传导矩阵，$\tau$是时间步长（即每一步的时间增量，$t_{n+1}=t_n + \tau$）。上式的解为：

$$L^{i+1} = \left( I - \tau \sum_{l=1}^m A_l (L^i) \right)^{-1} L^i \tag{11}$$

这种求解方法对任意时间步长$\tau$都有效和绝对稳定的。上式中$A_l$是对角占优的三对角线矩阵，这样的线性系统可以通过Thomas算法快速求解。

在这里简单解释下什么是对角占优的三对角线矩阵。首先，三对角线矩阵是指在矩阵中，除了主对角线及其上下相邻的两条对角线外，其余元素全为零的矩阵。比如：

$$
A = \begin{pmatrix}
b_1 & c_1 & 0 & \cdots & 0 \\
a_2 & b_2 & c_2 & \ddots & \vdots \\
0 & a_3 & b_3 & \ddots & 0 \\
\vdots & \ddots & \ddots & \ddots & c_{n-1} \\
0 & \cdots & 0 & a_n & b_n
\end{pmatrix} \tag{12}
$$

对角占优是指矩阵的每一行的主对角线元素的绝对值大于或等于该行其他元素绝对值之和。例如，考虑一个$4 \times 4$的对角占优的三对角线矩阵：

$$
A = \begin{pmatrix}
4 & 1 & 0 & 0 \\
1 & 3 & 1 & 0 \\
0 & 1 & 5 & 2 \\
0 & 0 & 2 & 6
\end{pmatrix} \tag{13}
$$

对角占优的三对角线矩阵必须是方阵。

>kaze详细实现请参见源码：[kaze](https://github.com/pablofdezalc/kaze)。

## 1.2.KAZE Features

### 1.2.1.Computation of the Nonlinear Scale Space

和[SIFT](http://shichaoxin.com/2022/12/29/OpenCV基础-第三十六课-SIFT特征检测/)类似，我们将非线性尺度空间也划分为$O$个octave，每个octave中包含$S$个sub-level。和[SIFT](http://shichaoxin.com/2022/12/29/OpenCV基础-第三十六课-SIFT特征检测/)不同的是，在非线性尺度空间中，我们不做下采样，一直保持原始图像的分辨率。在[SIFT](http://shichaoxin.com/2022/12/29/OpenCV基础-第三十六课-SIFT特征检测/)中，我们设置尺度$\sigma$为：

$$\sigma _i(o,s) = \sigma_0 2^{o+s/S}, \  o \in [0 ... O-1], \  s \in [0 ... S-1], \  i \in [0 ... N] \tag{14}$$

其中，$\sigma _0$是尺度参数的初始基准值，$N$是整个尺度空间中的图像总数$N = O * S$。在非线性尺度空间中，我们需要把以像素为单位的尺度参数$\sigma$转换为以时间为单位的尺度参数。转换公式为：

$$t_i = \frac{1}{2} \sigma_i^2, \  i = \{ 0 ... N \} \tag{15}$$

$t_i$被称为evolution time。值得注意的是，使用映射$\sigma_i \to t_i$只是为了从建立非线性尺度空间获得一组evolution time的值。通常，在非线性尺度空间每一个经过$t_i$滤波的结果图像与使用标准差为$\sigma_i$的高斯核对原始图像进行卷积所得的图像并不相符。但如果我们将传导函数$g$设为1（即$g$是一个常量函数）时，非线性尺度空间就符合了高斯尺度空间的意义。

给定输入图像，我们首先进行一次尺度为$\sigma_0$的高斯平滑，以减少噪声和图像伪影。然后，构建图像梯度直方图，获得对比度因子$k$。然后使用AOS方法，按照下式构建非线性尺度空间：

$$L^{i+1} = \left( I - (t_{i+1}-t_i) \cdot \sum_{l=1}^m A_l (L^i) \right)^{-1} L^i \tag{16}$$

下图展示了高斯尺度空间（Fig2上）和非线性尺度空间（Fig2下，使用$g_3$传导函数）的比较：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson44/44x2.png)

从Fig2可以看出，高斯尺度空间对图像中的所有结构进行平等的模糊，而非线性尺度空间则保证了强图像边缘不受影响。

### 1.2.2.Feature Detection

和[SIFT讲解](http://shichaoxin.com/2022/12/29/OpenCV基础-第三十六课-SIFT特征检测/)中的第1.2部分基本一样。

### 1.2.3.Feature Description

和[SURF讲解](http://shichaoxin.com/2023/08/18/论文阅读-SURF-Speeded-Up-Robust-Features/)中的第4部分基本一样。

# 2.AKAZE

AKAZE（Accelerated-KAZE）是KAZE算法的改进版，在加速KAZE算法的同时，保持了良好的特征检测和描述性能。

* 论文：Fast explicit diffusion for accelerated features in nonlinear scale spaces。
* 源码：[akaze](https://github.com/pablofdezalc/akaze)。

AKAZE使用了FED（Fast Explicit Diffusion）来显著加速非线性尺度空间中的特征检测。此外，AKAZE并没有使用[SURF描述符](http://shichaoxin.com/2023/08/18/论文阅读-SURF-Speeded-Up-Robust-Features/)，而是使用了M-LDB（Modified-Local Difference Binary）描述符。M-LDB描述符效率高，利用了非线性尺度空间中的梯度信息，具有尺度和旋转不变性，并且存储需求低。和BRISK、ORB、[SURF](http://shichaoxin.com/2023/08/18/论文阅读-SURF-Speeded-Up-Robust-Features/)、[SIFT](http://shichaoxin.com/2022/12/29/OpenCV基础-第三十六课-SIFT特征检测/)、KAZE等方法相比，AKAZE在速度和性能之间取得了良好的平衡。

# 3.OpenCV API

## 3.1.KAZE

```c++
Ptr<KAZE> create(
    bool extended=false, 
    bool upright=false,
    float threshold = 0.001f,
    int nOctaves = 4, 
    int nOctaveLayers = 4,
    KAZE::DiffusivityType diffusivity = KAZE::DIFF_PM_G2
    );
```

参数详解：

1. `extended`：是否将描述符扩展至128-byte。默认描述符为64-byte。同[cv::xfeatures2d::SURF::create](http://shichaoxin.com/2023/06/22/OpenCV基础-第三十九课-SURF特征检测/)。
2. `upright`：同[cv::xfeatures2d::SURF::create](http://shichaoxin.com/2023/06/22/OpenCV基础-第三十九课-SURF特征检测/)。
3. `threshold`：接受一个点的检测器响应阈值。
4. `nOctaves`：同[cv::xfeatures2d::SURF::create](http://shichaoxin.com/2023/06/22/OpenCV基础-第三十九课-SURF特征检测/)。
5. `nOctaveLayers`：同[cv::xfeatures2d::SURF::create](http://shichaoxin.com/2023/06/22/OpenCV基础-第三十九课-SURF特征检测/)。
6. `diffusivity`：传导函数$g$的形式，默认是$g_2$。可选的有：
    * `DIFF_PM_G1`：即$g_1$。
    * `DIFF_PM_G2`：即$g_2$。
    * `DIFF_WEICKERT`：即$g_3$。
    * `DIFF_CHARBONNIER`。

KAZE算法检测到的keypoints：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson44/44x3.png)

耗时50.019708ms。

## 3.2.AKAZE

```c++
Ptr<AKAZE> create(
    AKAZE::DescriptorType descriptor_type = AKAZE::DESCRIPTOR_MLDB,
    int descriptor_size = 0, 
    int descriptor_channels = 3,
    float threshold = 0.001f, 
    int nOctaves = 4,
    int nOctaveLayers = 4, 
    KAZE::DiffusivityType diffusivity = KAZE::DIFF_PM_G2,
    int max_points = -1
    );
```

参数详解：

1. `descriptor_type`：计算描述符的方法，默认为M-LDB。可选的有：
    * `DESCRIPTOR_KAZE`
    * `DESCRIPTOR_KAZE_UPRIGHT`
    * `DESCRIPTOR_MLDB`
    * `DESCRIPTOR_MLDB_UPRIGHT`
2. `descriptor_size`：描述符的尺寸（bits）。0表示全尺寸。
3. `descriptor_channels`：描述符的通道数（1，2，3）。
4. `threshold`：同KAZE API。
5. `nOctaves`：同KAZE API。
6. `nOctaveLayers`：同KAZE API。
7. `diffusivity`：同KAZE API。
8. `max_points`：返回点的最大数量。万一如果图像包含更多特征，则返回响应最高的特征。负值表示无限制。

AKAZE算法检测到的keypoints：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/OpenCVSeries/Lesson44/44x4.png)

耗时12.331333ms。确实比KAZE快了许多。

# 4.代码地址

1. [AKAZE局部匹配](https://github.com/x-jeff/OpenCV_Code_Demo/tree/master/Demo44)