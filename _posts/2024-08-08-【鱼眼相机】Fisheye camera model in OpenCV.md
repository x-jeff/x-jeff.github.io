---
layout:     post
title:      【鱼眼相机】Fisheye camera model in OpenCV
subtitle:   源码详解，cv::fisheye::calibrate，cv::fisheye::undistortPoints，cv::fisheye::estimateNewCameraMatrixForUndistortRectify，cv::fisheye::initUndistortRectifyMap
date:       2024-08-08
author:     x-jeff
header-img: blogimg/20211027.jpg
catalog: true
tags:
    - Camera Calibration
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.前言

OpenCV鱼眼相机模型部分所引用的论文：[A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses](https://shichaoxin.com/2024/07/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-A-Generic-Camera-Model-and-Calibration-Method-for-Conventional,-Wide-Angle,-and-Fish-Eye-Lenses/)。

[OpenCV官方文档](https://docs.opencv.org/4.6.0/db/d58/group__calib3d__fisheye.html#gad626a78de2b1dae7489e152a5a5a89e1)（本人所用OpenCV版本为4.6.0）的描述如下。

从世界坐标系到相机坐标系的转换表示为：

$$\mathbf{X}c = \mathbf{RX}+\mathbf{T} \tag{1}$$

其中，$\mathbf{R}$是旋转矩阵，$\mathbf{T}$是平移向量，$\mathbf{X}$是世界坐标系中的点。$\mathbf{X}c$是相机坐标系下的点，其$x,y,z$坐标分别表示为：

$$x = \mathbf{X}c_1 \tag{2}$$

$$y = \mathbf{X}c_2 \tag{3}$$

$$z = \mathbf{X}c_3 \tag{4}$$

如下图所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/CameraCalibration/20240808/1.png)

设位于相机坐标系中的点$P$的坐标为$(x,y,z)$。如果使用[perspective projection](https://shichaoxin.com/2024/07/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-A-Generic-Camera-Model-and-Calibration-Method-for-Conventional,-Wide-Angle,-and-Fish-Eye-Lenses/#2aradially-symmetric-model)模型，则点$P$在图像坐标系中的投影点为$p'$，其坐标为$(a,b)$，根据相似三角形，我们可以得到：

$$\frac{x}{a} = \frac{y}{b} = \frac{z}{f} \tag{5}$$

考虑在归一化成像平面（即$f=1$时的成像平面）做畸变校正，可得：

$$a = \frac{x}{z} \cdot f = \frac{x}{z} \tag{6}$$

$$b = \frac{y}{z} \cdot f = \frac{y}{z} \tag{7}$$

$p'$到原点的半径$r$为：

$$r^2 = a^2 + b^2 \tag{8}$$

在[perspective projection](https://shichaoxin.com/2024/07/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-A-Generic-Camera-Model-and-Calibration-Method-for-Conventional,-Wide-Angle,-and-Fish-Eye-Lenses/#2aradially-symmetric-model)模型中，入射角等于出射角，所以有：

$$\tan \theta = \frac{r}{f} = r \tag{9}$$

即：

$$\theta = atan (\frac{r}{f}) = atan (r) \tag{10}$$

根据[论文](https://shichaoxin.com/2024/07/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-A-Generic-Camera-Model-and-Calibration-Method-for-Conventional,-Wide-Angle,-and-Fish-Eye-Lenses/)提出的通用相机模型，点$p$到原点的半径$\theta_d$可计算为：

$$\theta_d = \theta (1 + k_1\theta^2 + k_2 \theta^4+k_3\theta^6 +k_4\theta^8) \tag{11}$$

>和[论文中的式(6)](https://shichaoxin.com/2024/07/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-A-Generic-Camera-Model-and-Calibration-Method-for-Conventional,-Wide-Angle,-and-Fish-Eye-Lenses/)相比，这里相当于是把$k_1$置为1了。

据此，我们可以计算点$p$的坐标$(x',y')$为：

$$x' = \frac{\theta_d}{r} \cdot a \tag{12}$$

$$y' = \frac{\theta_d}{r} \cdot b \tag{13}$$

>注意，这里的$p$坐标是在图像坐标系（归一化成像平面）下，假设其对应的相机坐标系下的点为$(Xc_p,Yc_p,Zc_p)$，因为有$f=1$，所以$x' = \frac{Xc_p}{Zc_p},y'=\frac{Yc_p}{Zc_p}$。因此$(x',y')$可以直接使用内参矩阵转换为像素坐标系（实际成像平面）下的点（见下面式(14)和式(15)）。

最终，得到点$p$在像素坐标系下的坐标：

$$u = f_x(x'+\alpha y')+c_x \tag{14}$$

$$v = f_y y' + c_y \tag{15}$$

其中，$\alpha$是[倾斜因子](https://shichaoxin.com/2022/12/16/%E7%9B%B8%E6%9C%BA%E6%A0%87%E5%AE%9A-%E5%BC%A0%E6%AD%A3%E5%8F%8B%E6%A0%87%E5%AE%9A%E6%B3%95/#2%E5%80%BE%E6%96%9C%E5%9B%A0%E5%AD%90)。

>这里OpenCV其实用的是[论文](https://shichaoxin.com/2024/07/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-A-Generic-Camera-Model-and-Calibration-Method-for-Conventional,-Wide-Angle,-and-Fish-Eye-Lenses/)中的$\mathbf{p}_9$模型。在OpenCV中，所谓的鱼眼镜头去畸变过程就是对投影模型的转换（从点$p$纠正到点$p'$），而并没有真正做[论文](https://shichaoxin.com/2024/07/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-A-Generic-Camera-Model-and-Calibration-Method-for-Conventional,-Wide-Angle,-and-Fish-Eye-Lenses/)第2.B部分提到的畸变校正。

## 1.1.相机分辨率对相机内参的影响

一些题外话，如果使用不同的图像分辨率，其对应的内参矩阵也是不同的。假设在分辨率$w \times h$下，图像坐标系到像素坐标系的转换为：

$$u = f_x x' + c_x \tag{16}$$

$$v = f_y y' + c_y \tag{17}$$

>焦距$f$的单位是mm，$dx$表示一个像素在$x$方向的长度是多少mm，所以$f_x$可以视为在$x$方向的焦距，只不过单位是像素。$f_y$同理。

如果我们将分辨率调整为$w' \times h'$，则上述转换对应变为：

$$\frac{w'}{w}u = \frac{w'}{w} (f_x x' + c_x) \tag{18}$$

$$\frac{h'}{h}v = \frac{h'}{h} (f_y y' + c_y) \tag{19}$$

对应的内参矩阵调整为：

$$\begin{bmatrix} \frac{w'}{w} \cdot f_x & 0 & \frac{w'}{w} \cdot c_x \\ 0 & \frac{h'}{h} \cdot f_y & \frac{h'}{h} \cdot c_y \\ 0 & 0 & 1 \\   \end{bmatrix} \tag{20}$$

# 2.`cv::fisheye::calibrate`

```c++
double cv::fisheye::calibrate (
    InputArrayOfArrays objectPoints
    InputArrayOfArrays imagePoints,
    const Size& image_size,
    InputOutputArray K,
    InputOutputArray D,
    OutputArrayOfArrays rvecs,
    OutputArrayOfArrays tvecs,
    int flags = 0,
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, DBL_EPSILON) 
)
```

参数解释：

1. `objectPoints`：世界坐标系下的点。
2. `imagePoints`：像素坐标系下的点，和`objectPoints`中的点一一对应。
3. `image_size`：图像大小，仅用于初始化相机内参矩阵。
4. `K`：输出的$3\times 3$浮点型的相机内参矩阵：
   
   $$A = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

   如果设置了`fisheye::CALIB_USE_INTRINSIC_GUESS`，在调用函数之前，$f_x,f_y,c_x,c_y$等值必须被指定初始化。
5. `D`：输出的畸变参数$(k_1,k_2,k_3,k_4)$，见式(11)。
6. `rvecs`：对于每个校正板图像，都输出一个旋转向量（Rodrigues公式，其可以将旋转向量转换为旋转矩阵）。
7. `tvecs`：对于每个校正板图像，都输出一个平移向量。
8. `flags`：参数设置：
   * `fisheye::CALIB_USE_INTRINSIC_GUESS`：相机矩阵中包含了$f_x,f_y,c_x,c_y$的有效初始值，这些值会被进一步优化。否则，$(c_x,c_y)$的初始值被设置为图像的中心，焦距则通过最小二乘法计算。
   * `fisheye::CALIB_RECOMPUTE_EXTRINSIC`：在每次内参优化迭代之后都会重新计算外参。
   * `fisheye::CALIB_CHECK_COND`：这些函数将检查条件编号的有效性。
   * `fisheye::CALIB_FIX_SKEW`：[倾斜因子](https://shichaoxin.com/2022/12/16/%E7%9B%B8%E6%9C%BA%E6%A0%87%E5%AE%9A-%E5%BC%A0%E6%AD%A3%E5%8F%8B%E6%A0%87%E5%AE%9A%E6%B3%95/#2%E5%80%BE%E6%96%9C%E5%9B%A0%E5%AD%90)（alpha）设为0并保持为0。
   * `fisheye::CALIB_FIX_K1`,...,`fisheye::CALIB_FIX_K4`：将特定的畸变系数设为0并保持为0。
   * `fisheye::CALIB_FIX_PRINCIPAL_POINT`：在全局优化过程中主点$(c_x,c_y)$不会改变。它会保持在图像中心或`fisheye::CALIB_USE_INTRINSIC_GUESS`中的预设值。
   * `fisheye::CALIB_FIX_FOCAL_LENGTH`：在全局优化过程中焦距不会改变。它是$\max(width,height)/\pi$或在`fisheye::CALIB_USE_INTRINSIC_GUESS`中预设的$f_x,f_y$。
9. `criteria`：迭代优化算法的停止条件。

# 3.`cv::fisheye::undistortPoints`

## 3.1.API介绍

```c++
void cv::fisheye::undistortPoints (
    InputArray distorted,
    OutputArray undistorted,
    InputArray K,
    InputArray D,
    InputArray R = noArray(),
    InputArray P = noArray(),
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 10, 1e-8) 
)
```

参数解释：

1. `distorted`：带有畸变的2D点（像素坐标系）。
2. `undistorted`：去畸变后的2D点（像素坐标系）。
3. `K`：内参矩阵。
4. `D`：畸变系数$(k_1,k_2,k_3,k_4)$。
5. `R`：校正变换。
6. `P`：新的相机内参矩阵（$3 \times 3$）或新的投影矩阵（$3 \times 4$）。
7. `criteria`：迭代优化算法的停止条件。

## 3.2.源码解析

首先是对参数有效性的判断：

```c++
CV_INSTRUMENT_REGION();

// will support only 2-channel data now for points
CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
undistorted.create(distorted.size(), distorted.type());

CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
CV_Assert(D.total() == 4 && K.size() == Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

CV_Assert(criteria.isValid());
```

然后读入内参矩阵和畸变系数，将$f_x,f_y$存入`f`，$c_x,c_y$存入`c`，$(k_1,k_2,k_3,k_4)$存入`k`：

```c++
cv::Vec2d f, c;
if (K.depth() == CV_32F)
{
    Matx33f camMat = K.getMat();
    f = Vec2f(camMat(0, 0), camMat(1, 1));
    c = Vec2f(camMat(0, 2), camMat(1, 2));
}
else
{
    Matx33d camMat = K.getMat();
    f = Vec2d(camMat(0, 0), camMat(1, 1));
    c = Vec2d(camMat(0, 2), camMat(1, 2));
}

Vec4d k = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();
```

接着读入参数`R`和`P`，并将两个矩阵的乘积存入`RR`（后面会解释这两个矩阵的作用）：

```c++
cv::Matx33d RR = cv::Matx33d::eye();
if (!R.empty() && R.total() * R.channels() == 3)
{
    cv::Vec3d rvec;
    R.getMat().convertTo(rvec, CV_64F);
    RR = cv::Affine3d(rvec).rotation();
}
else if (!R.empty() && R.size() == Size(3, 3))
    R.getMat().convertTo(RR, CV_64F);

if(!P.empty())
{
    cv::Matx33d PP;
    P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
    RR = PP * RR;
}
```

如果点坐标是float型，则带畸变的点存到`srcf`，去畸变的点存到`dstf`；如果点坐标是double型，则带畸变的点存到`srcd`，去畸变的点存到`dstd`。`n`是点的个数。

```c++
// start undistorting
const cv::Vec2f* srcf = distorted.getMat().ptr<cv::Vec2f>();
const cv::Vec2d* srcd = distorted.getMat().ptr<cv::Vec2d>();
cv::Vec2f* dstf = undistorted.getMat().ptr<cv::Vec2f>();
cv::Vec2d* dstd = undistorted.getMat().ptr<cv::Vec2d>();

size_t n = distorted.total();
int sdepth = distorted.depth();
```

读入优化算法的终止条件：

```c++
const bool isEps = (criteria.type & TermCriteria::EPS) != 0;

/* Define max count for solver iterations */
int maxCount = std::numeric_limits<int>::max();
if (criteria.type & TermCriteria::MAX_ITER) {
    maxCount = criteria.maxCount;
}
```

接下来是算法的主体部分，对每个点依次进行去畸变：

```c++
for(size_t i = 0; i < n; i++ )
{
    Vec2d pi = sdepth == CV_32F ? (Vec2d)srcf[i] : srcd[i];  // image point
    Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point

    double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);

    // the current camera model is only valid up to 180 FOV
    // for larger FOV the loop below does not converge
    // clip values so we still get plausible results for super fisheye images > 180 grad
    theta_d = min(max(-CV_PI/2., theta_d), CV_PI/2.);

    bool converged = false;
    double theta = theta_d;

    double scale = 0.0;

    if (!isEps || fabs(theta_d) > criteria.epsilon)
    {
        // compensate distortion iteratively using Newton method

        for (int j = 0; j < maxCount; j++)
        {
            double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
            double k0_theta2 = k[0] * theta2, k1_theta4 = k[1] * theta4, k2_theta6 = k[2] * theta6, k3_theta8 = k[3] * theta8;
            /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
            double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
            theta = theta - theta_fix;

            if (isEps && (fabs(theta_fix) < criteria.epsilon))
            {
                converged = true;
                break;
            }
        }

        scale = std::tan(theta) / theta_d;
    }
    else
    {
        converged = true;
    }

    // theta is monotonously increasing or decreasing depending on the sign of theta
    // if theta has flipped, it might converge due to symmetry but on the opposite of the camera center
    // so we can check whether theta has changed the sign during the optimization
    bool theta_flipped = ((theta_d < 0 && theta > 0) || (theta_d > 0 && theta < 0));

    if ((converged || !isEps) && !theta_flipped)
    {
        Vec2d pu = pw * scale; //undistorted point

        // reproject
        Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
        Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final

        if( sdepth == CV_32F )
            dstf[i] = fi;
        else
            dstd[i] = fi;
    }
    else
    {
        // Vec2d fi(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
        Vec2d fi(-1000000.0, -1000000.0);

        if( sdepth == CV_32F )
            dstf[i] = fi;
        else
            dstd[i] = fi;
    }
}
```

我们来依次讲解每一步处理的作用。首先，`pi`存的是像素坐标系下带有畸变的点，通过内参矩阵（此处不考虑[倾斜因子](https://shichaoxin.com/2022/12/16/%E7%9B%B8%E6%9C%BA%E6%A0%87%E5%AE%9A-%E5%BC%A0%E6%AD%A3%E5%8F%8B%E6%A0%87%E5%AE%9A%E6%B3%95/#2%E5%80%BE%E6%96%9C%E5%9B%A0%E5%AD%90)），得到点`pw`，即式(14)和式(15)的反向计算，得到$x',y'$。接着计算`theta_d`，即$\theta_d = \sqrt{x'^2+y'^2}$。

现在我们已经有了$\theta_d$，我们的目标是求解$\theta$。构建如下式子：

$$f(\theta) = \theta (1 + k_1\theta^2 + k_2 \theta^4+k_3\theta^6 +k_4\theta^8) - \theta_d \tag{21}$$

我们的目标就是最小化上式。我们设$\theta_0 = \theta_d$，使用牛顿迭代法求解：

$$\theta_{n+1} = \theta_n - \frac{f(\theta_n)}{f'(\theta_n)} \tag{22}$$

以上便是`for (int j = 0; j < maxCount; j++)`循环里的计算。接下来计算的`scale`就是$\frac{r}{\theta_d}$。通过`Vec2d pu = pw * scale;`计算得到的`pu`就是$(a,b)$（见式(12)和式(13)）。`pu`的齐次坐标可视为相机坐标系下的点，然后通过`RR`矩阵进行转换（即`Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0);`），这个`RR`由两个矩阵相乘得到：

* 第一个矩阵是一个旋转矩阵（即输入参数`R`），将相机坐标系下的该点进行旋转，默认是单位矩阵，不进行任何旋转。
* 第二个矩阵是一个新的内参矩阵（即输入参数`P`），用于将相机坐标系下的该点转换回像素坐标系。默认也可视为是一个单位矩阵，即不转换回像素坐标系。

最后将得到的`pr`进行归一化：`Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);`，得到像素坐标系下去畸变之后的点`fi`，至此，针对一个点的去畸变操作完成。最后将所有去畸变的点存入`dstf`或`dstd`，并通过API返回。

# 4.`cv::fisheye::estimateNewCameraMatrixForUndistortRectify`

## 4.1.API介绍

```c++
void cv::fisheye::estimateNewCameraMatrixForUndistortRectify (
    InputArray K,
    InputArray D,
    const Size & image_size,
    InputArray R,
    OutputArray P,
    double balance = 0.0,
    const Size & new_size = Size(),
    double fov_scale = 1.0 
)
```

参数解释：

1. `K`：输入的带有畸变的内参矩阵。
2. `D`：输入的畸变系数$(k_1,k_2,k_3,k_4)$。
3. `image_size`：图像大小。
4. `R`：校正变换。
5. `P`：输出的去畸变后的内参矩阵。
6. `balance`：在最小焦距和最大焦距范围内设置一个新的焦距。该参数取值范围为$[0,1]$。
7. `new_size`：新的图像大小。
8. `fov_scale`：新焦距的除数。

## 4.2.源码解析

对输入参数有效性的判断：

```c++
CV_INSTRUMENT_REGION();

CV_Assert( K.size() == Size(3, 3)       && (K.depth() == CV_32F || K.depth() == CV_64F));
CV_Assert(D.empty() || ((D.total() == 4) && (D.depth() == CV_32F || D.depth() == CV_64F)));
```

输入图像尺寸`(w,h)`和`balance`：

```c++
int w = image_size.width, h = image_size.height;
balance = std::min(std::max(balance, 0.0), 1.0);
```

将图像上下左右4个边界上的中心点喂入`cv::fisheye::undistortPoints`，得到去畸变后的点：

```c++
cv::Mat points(1, 4, CV_64FC2);
Vec2d* pptr = points.ptr<Vec2d>();
pptr[0] = Vec2d(w/2, 0);
pptr[1] = Vec2d(w, h/2);
pptr[2] = Vec2d(w/2, h);
pptr[3] = Vec2d(0, h/2);

fisheye::undistortPoints(points, points, K, D, R);
```

输入参数`R`只用于传给`cv::fisheye::undistortPoints`，`R`默认是单位矩阵，不做任何旋转。而我们并没有给`cv::fisheye::undistortPoints`传入参数`P`，相当于我们没有将点转回到像素坐标系，所以`cv::fisheye::undistortPoints`返回的`points`是相机坐标系下的齐次坐标。

将`points`中4个点的中心存入`cn`：

```c++
cv::Scalar center_mass = mean(points);
cv::Vec2d cn(center_mass.val);
```

对`points`和`cn`的x方向和y方向做各向同性处理：

```c++
double aspect_ratio = (K.depth() == CV_32F) ? K.getMat().at<float >(0,0)/K.getMat().at<float> (1,1)
                                            : K.getMat().at<double>(0,0)/K.getMat().at<double>(1,1);

// convert to identity ratio
cn[1] *= aspect_ratio;
for(size_t i = 0; i < points.total(); ++i)
    pptr[i][1] *= aspect_ratio;
```

求`points`中4个点x坐标的最小值`minx`、x坐标的最大值`maxx`、y坐标的最小值`miny`以及y坐标的最大值`maxy`：

```c++
double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
for(size_t i = 0; i < points.total(); ++i)
{
    miny = std::min(miny, pptr[i][1]);
    maxy = std::max(maxy, pptr[i][1]);
    minx = std::min(minx, pptr[i][0]);
    maxx = std::max(maxx, pptr[i][0]);
}
```

接下来就是去畸变内参矩阵中焦距（即$f_x,f_y$）的计算：

```c++
double f1 = w * 0.5/(cn[0] - minx);
double f2 = w * 0.5/(maxx - cn[0]);
double f3 = h * 0.5 * aspect_ratio/(cn[1] - miny);
double f4 = h * 0.5 * aspect_ratio/(maxy - cn[1]);

double fmin = std::min(f1, std::min(f2, std::min(f3, f4)));
double fmax = std::max(f1, std::max(f2, std::max(f3, f4)));

double f = balance * fmin + (1.0 - balance) * fmax;
f *= fov_scale > 0 ? 1.0/fov_scale : 1.0;
```

为了确保去畸变后的图像在新的图像平面上具有适当的视场，并且图像的所有部分都能够正确投影到新的图像平面上，这里将`f`限定在`fmin`和`fmax`范围内。然后再用`fov_scale`进一步调整焦距。

这里先解释下`f1`的计算（`f2`、`f3`、`f4`同理）。我们希望去畸变图像的大小和原始图像差不多，即中心点到边界的距离是差不多的，也就是说，`cn[0]`和`minx`转换回像素坐标系后的距离和`w * 0.5`是一样的。`cn[0]`转换回像素坐标系的公式是`cn[0] * f1 + cx`，`minx`转换回像素坐标系的公式是`minx * f1 + cx`，二者的距离是`(cn[0]-minx) * f1`，这个距离要等于`w * 0.5`，据此，我们便可得到上述代码中`f1`的计算。

通过对齐中心点到不同边界的距离，我们可以得到`f1`、`f2`、`f3`、`f4`，这4个焦距的最小值记为`fmin`，最大值记为`fmax`，通过比例参数`balance`来设置`f`在这个范围内的取值。`fov_scale`用于对焦距的进一步调整。

去畸变内参矩阵中$c_x,c_y$的计算：

```c++
 cv::Vec2d new_f(f, f), new_c = -cn * f + Vec2d(w, h * aspect_ratio) * 0.5;
```

我们希望中心点`cn`可以对齐原始图像的中心位置，所以会有（以x坐标为例）`cn[0] * f + cx = w * 0.5`，因此，可以得到`cx = -cn[0] * f + w * 0.5`，`cy`同理。

恢复各向异性：

```c++
// restore aspect ratio
new_f[1] /= aspect_ratio;
new_c[1] /= aspect_ratio;
```

如果去畸变图像的大小不想和原始图像一样，可以设置新的图像大小（见本文第1.1部分）：

```c++
if (!new_size.empty())
{
    double rx = new_size.width /(double)image_size.width;
    double ry = new_size.height/(double)image_size.height;

    new_f[0] *= rx;  new_f[1] *= ry;
    new_c[0] *= rx;  new_c[1] *= ry;
}
```

返回最终的去畸变内参矩阵：

```c++
Mat(Matx33d(new_f[0], 0, new_c[0],
            0, new_f[1], new_c[1],
            0,        0,       1)).convertTo(P, P.empty() ? K.type() : P.type());
```

# 5.`cv::fisheye::initUndistortRectifyMap`

有了去畸变的内参矩阵之后，如何得到去畸变后的图像呢？我们可以调用`cv::fisheye::initUndistortRectifyMap`先得到去畸变图像的`map1`和`map2`，然后再调用[`cv::remap`](https://shichaoxin.com/2021/06/29/OpenCV%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%BA%8C%E5%8D%81%E8%AF%BE-%E5%83%8F%E7%B4%A0%E9%87%8D%E6%98%A0%E5%B0%84/)得到去畸变后的图像。

```c++
void cv::fisheye::initUndistortRectifyMap (
    InputArray K,
    InputArray D,
    InputArray R,
    InputArray P,
    const cv::Size & size,
    int m1type,
    OutputArray map1,
    OutputArray map2 
)
```

参数解释：

1. `K`：带有畸变的内参矩阵。
2. `D`：畸变系数$(k_1,k_2,k_3,k_4)$。
3. `R`：校正变换。
4. `P`：去畸变的内参矩阵。
5. `size`：去畸变图像的大小。
6. `m1type`：`map1`的类型，可以是`CV_32FC1`或`CV_16SC2`。
7. `map1`：第一个输出map，见[`cv::remap`](https://shichaoxin.com/2021/06/29/OpenCV%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%BA%8C%E5%8D%81%E8%AF%BE-%E5%83%8F%E7%B4%A0%E9%87%8D%E6%98%A0%E5%B0%84/)。
8. `map2`：第二个输出map，见[`cv::remap`](https://shichaoxin.com/2021/06/29/OpenCV%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%BA%8C%E5%8D%81%E8%AF%BE-%E5%83%8F%E7%B4%A0%E9%87%8D%E6%98%A0%E5%B0%84/)。

# 6.参考资料

1. [鱼眼镜头的成像原理到畸变矫正（完整版）](https://blog.csdn.net/qq_16137569/article/details/112398976)
2. [深入洞察OpenCV鱼眼模型之成像投影和畸变表估计系数相互转化](https://zhuanlan.zhihu.com/p/655174655)