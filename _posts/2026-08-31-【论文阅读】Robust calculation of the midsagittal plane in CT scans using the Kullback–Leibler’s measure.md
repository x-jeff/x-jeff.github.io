---
layout:     post
title:      【论文阅读】Robust calculation of the midsagittal plane in CT scans using the Kullback–Leibler’s measure
subtitle:   Midsagittal plane，CT，Symmetry，Stroke
date:       2026-08-31
author:     x-jeff
header-img: blogimg/20221110.jpg
catalog: true
tags:
    - Medical Imaging
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

**大脑半球间裂（the Interhemispheric Fissure，IF）**，也称**大脑纵裂（the longitudinal cerebral fissure）**，是人脑中重要的解剖标志之一。通常使用**正中矢状面（the MidSagittal Plane，MSP）**来近似表示半球间裂。MSP将大脑分为左右两个半球，其可作为后续脑部图像分析的基础，比如左右对称比较、脑图谱配准、卒中侧别判断等。

大脑纵裂动图示意：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/1.gif)

作者在之前的研究中，就提出了两种基于局部对称性和[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)来计算MSP的方法，这两种方法都在MR图像上得到了有效的验证，并且被集成到作者开发的MR卒中计算机辅助诊断系统（MR stroke CAD system）中。该系统已经提供给全球多家医院和公司使用。

上述提到的算法有2个限制：1）预先假设MSP的方向应当接近于一个垂直平面，因此，当患者头部倾斜超过一定角度时，算法就可能失效；2）在CT图像上的验证不足。

在急诊情况下，CT扫描时，患者的头部往往处于任意姿势，头部可能同时在三个方向上发生较大的倾斜或旋转：

* Yaw：在axial平面旋转。
* Roll：在coronal平面旋转。
* Pitch：在sagittal平面旋转。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/2.png)

此外，由于CT的层厚较大以及存在明显的部分容积效应（partial volume effect），脑部结构在连续层中可能并不能保持连续可见。即使只是很小的头部倾斜，也可能导致某个脑结构在大脑一侧能够看到，而在另一侧看不到，或者同一个解剖结构在左右两个半球中呈现出不同的影像表现。CT图像所具有的这些典型特性，使得设计一个鲁棒的针对CT数据的MSP检测算法变得具有挑战性。

>部分容积效应可简单的理解为：一个CT体素里同时包含了多种不同组织，最终这个体素只能给出一个CT值，因此这个CT值会变成这些组织的“混合结果”。例如，一个体素中如果50%是脑组织，CT值约为40HU，另外50%是脑脊液，CT值约为10HU，那么这个体素的CT值最终可能是$0.5 \times 40 + 0.5 \times 10 = 25$HU。
>
>层厚越大，部分容积效应越严重。

# 2.Materials and methods

## 2.1.Materials

研究共使用了208组CT数据，既包含正常数据，也包含患病数据。数据的层厚范围为$1.5 \sim 6$mm，其中95%的数据层厚为5mm或者更大。在axial平面内，像素尺寸范围为$0.3906 \sim 0.5547$mm。对于所有数据，扫描范围均覆盖整个头部。坐标轴的定义如下：

* X轴：左右方向（left-right）。
* Y轴：前后方向（anterior-posterior）。
* Z轴：上下方向（inferior-superior）。

## 2.2.Method

### 2.2.1.Algorithm

[论文“Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure”](https://shichaoxin.com/2026/08/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Extraction-of-the-midsagittal-plane-from-morphological-neuroimages-using-the-Kullback-Leibler-s-measure/)假设：MSP的方向接近于垂直平面，或者患者头部仅存在较小程度的倾斜。因此，在患者头部发生较大倾斜的情况下，该算法可能无法正确识别MSP。对于这类情况，就需要先对CT扫描数据进行旋转，使得MSP在axial平面上的投影方向大致接近垂直方向。所需的旋转角度主要通过两种信息来估计：1）在axial图像上对颅骨轮廓进行椭圆拟合；2）估计头部的左右对称性。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/3.png)

在Fig1中，将MSP投影到某个axial平面得到的线称为**MSL（MidSagittal Line）**。在颅骨前后方向（即Y方向）跨度最大的那一层axial图像中，对颅骨外边缘进行椭圆拟合，拟合椭圆的长轴记为$M_j$。MSL和$M_j$的夹角记为$\theta_d$。

对于每张axial图像，以MSL为分界线，把脑组织区域分成两部分：1）$\text{brain}_1$表示MSL左侧的脑组织区域；2）$\text{brain}_2$表示MSL右侧的脑组织区域。将所有axial图像的$\text{brain}_1$面积和$\text{brain}_2$面积各自累加，累加的$\text{brain}_1$面积之和除以$\text{brain}_2$面积之和，比值记为$lr\\_ratio$。

$\theta_d$和$lr\\_ratio$被用来衡量MSP的偏离程度。

算法整体流程见下图：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/4.png)

* 使用[论文“Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure”](https://shichaoxin.com/2026/08/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Extraction-of-the-midsagittal-plane-from-morphological-neuroimages-using-the-Kullback-Leibler-s-measure/)提出的算法在原始数据上检测$\text{MSP}_1$。
* 基于$\text{MSP}_1$，计算$\theta_d$：
    * 颅骨提取。
    * 对颅骨外轮廓进行椭圆拟合。
    * 计算$\theta_d$。
* 基于$\text{MSP}_1$，计算$lr\\_ratio$：
    * 脑组织提取。
    * 左右脑组织分别计算面积。
    * 计算$lr\\_ratio$。
* 是否满足$\theta_d > 7.8 °$或$lr\\_ratio > 10%$？
    * 如果满足：
        * 对原始数据进行旋转。
        * 使用[论文“Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure”](https://shichaoxin.com/2026/08/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Extraction-of-the-midsagittal-plane-from-morphological-neuroimages-using-the-Kullback-Leibler-s-measure/)提出的算法在旋转后的数据上检测得到$\text{MSP}_2$。
        * 将$\text{MSP}_2$映射回原始数据中，得到$\text{MSP}_3$。
        * $\text{MSP}_{\text{final}} = \text{MSP}_3$。
    * 如果不满足：
        * $\text{MSP}_{\text{final}} = \text{MSP}_3$。
* $\text{MSP}_{\text{final}}$就是最终检测到的MSP。

### 2.2.2.MSP calculation with KL-measure

设$p = \\{ p_i \\}$和$q = \\{ q_i \\}$为两个离散概率分布，其中$p_i$和$q_i$分别表示第$i$个状态在各自分布中出现的概率。则[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)可定义为：

$$\begin{align*} I(p/q) &= \sum_i p_i \log (p_i) - \sum_i p_i \log (q_i) \\&= \sum_i p_i \log (p_i / q_i) \end{align*} \tag{1}$$

基于[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)的MSP检测算法可见：[【论文阅读】Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure](https://shichaoxin.com/2026/08/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Extraction-of-the-midsagittal-plane-from-morphological-neuroimages-using-the-Kullback-Leibler-s-measure/)。

### 2.2.3.Calculation of $\theta_d$

将顺时针方向规定为角度的正方向。定义$\theta_{\text{MSP}}$为MSL与Y轴之间的夹角，$\theta_{\text{ellipse}}$为$M_j$与Y轴之间的夹角。因此，$\theta_d$可定义为：$\theta_d = \lvert \theta_{\text{MSP}} - \theta_{\text{ellipse}} \rvert$。其中，$\theta_{\text{MSP}}$的计算比较直接，在任意axial图像上，可通过MSL求出。而$\theta_{\text{ellipse}}$的计算则需要以下几个步骤：

1. 首先需要确定一张axial图像，记为$slice\\_ref$，并在这张axial图像上拟合颅骨椭圆。寻找$slice\\_ref$的标准是：前后方向（anterior–posterior）跨度最大的axial图像。作者认为这样找到的$slice\\_ref$可近似对应于经过前连合（anterior commissure，AC）的axial图像。寻找$slice\\_ref$的具体步骤见下：
    * 首先初始化一个VOI，这个VOI从切片$s$开始，一直到最靠近头顶部的切片。其中这个$s$是近似对应AC平面的位置，其计算公式为：$s = round (\frac{43}{43 + 74} \times \text{数据总层数})$。43和74的单位都是mm，值来自Talairach–Tournoux atlas。43mm是从AC到颞叶皮层最下端点（CT颅脑扫描大约都是从这里开始的）的距离。74mm是从AC到大脑上方解剖标志点的距离。因此，这两个值被用来估计AC在数据场中的位置。此外，根据Talairach–Tournoux atlas，经过AC的这个axial图像具有最大的前后方向跨度。因此，在这一层，颅骨应该具有最大的偏心程度，并且拟合椭圆的长轴方向应该大致对应于IF的方向。个人注解：如果颅脑扫描的范围过大或者过小，这种寻找$s$的方法可能会有问题。
    * 对于VOI中的每一张axial图像，基于颅骨CT值对颅骨进行分割，低阈值设为350HU，高阈值设为2250HU，从而得到二值化的颅骨区域。阈值的选择依据为，在CT图像中，骨组织的CT值范围约为1000-2250HU，但考虑到部分容积效应，将下限降为350HU。
    * 对于VOI中的每一张axial图像，提取其中的最大连通域，计算该连通域在前后方向上的跨度。
    * 找到前后方向跨度最大的那张切片，即为$slice\\_ref$。
2. 基于提取颅骨的CT阈值，对$slice\\_ref$重新提取颅骨，并找到最大颅骨连通域。
3. 对第2步得到的颅骨连通域的外轮廓进行椭圆拟合。
4. 基于第3步得到的拟合椭圆，计算$\theta_{\text{ellipse}}$。

### 2.2.4.Calculation of $lr\\_ratio$

1. 首先，对三维体数据进行阈值分割，用于提取脑组织。阈值可根据窗宽、窗位来设置。通常，下阈值设为-5HU，上阈值设为75HU。在得到的二值体数据中，白色体素表示脑组织，黑色体素表示非脑组织。
2. 接下来，分析每一个脑组织体素相对于$MSP_1$的位置。设$MSP_1$的平面方程为$Ax+By+Cz+D=0$且$A>0$，其中$x,y,z$是三维体数据的坐标。对于二值体数据中任意一个白色体素，其坐标记为$(v_x,v_y,v_z)$，计算$f(v_x,v_y,v_z) = A v_x + B v_y + C v_z + D$。如果$f(v_x,v_y,v_z) > 0$，则表示该体素位于$MSP_1$的左侧；如果$f(v_x,v_y,v_z) < 0$，则表示该体素位于$MSP_1$的右侧。位于$MSP_1$左侧的所有白色体素的数量记为$V_{left}$；位于$MSP_1$右侧的所有白色体素的数量记为$V_{right}$。最终有：$lr\\\_ratio = \frac{\lvert V_{right} - V_{left} \rvert}{ \max (V_{right},V_{left}) }$。

### 2.2.5.Volume rotation

体数据按照如下方式进行旋转。首先，计算二值化后切片$slice\\\_ref$的质心，其坐标记为$(X_c,Y_c,Z_{slice\\\_ref})$。然后，将这个质心投影到每一张axial切片上，并将投影点作为对应切片的旋转中心。例如，对于第5张axial切片，它的旋转中心就是$(X_c,Y_c,5)$。接下来，将每一张axial切片旋转$\theta_{\text{ellipse}}$。旋转后的切片采用双线性插值来进行重建。这个旋转操作相当于是整个三维数据，以经过$slice\\_ref$质心且垂直于axial平面的直线为旋转轴。

### 2.2.6.Analysis

将全部数据随机划分为8个大小相等的子集，每个子集轮流做测试集，另外7个子集为训练集。

对于定量分析，GT由神经解剖学专家人工标注，其在每一张axial切片上绘制GTL（Ground Truth Line），让GTL尽可能从IF中央穿过。

一共使用3个定量分析指标：

1. 角度偏差$\alpha$，单位为度。$\alpha$为MSP与每一层GTL之间绝对夹角的平均值。
2. 距离偏差$d$，单位为mm。作者把每一条GTL线段的两个端点放在脑组织bounding box的边界上，这样做是为了能捕获可能出现的最大误差。计算GTL两个端点到MSP的欧氏距离，最后对这些距离取平均，得到$d$。
3. 非平面性指标$\theta_{GT}$，是一个角度指标，单位为度。$\theta_{GT}$定义为所有GTL之间的最大夹角，这个指标用于描述IF的弯曲程度。

此外，作者还使用一组高分辨率CT数据来验证该方法对于头部旋转的鲁棒性。这组CT数据的层厚为1.5mm。之后，分别在yaw、pitch、roll三个方向上人为的旋转体数据，并使用三线性插值生成旋转后的三维数据。三个方向的旋转步长都是5°，yaw方向的旋转范围为顺时针5°到40°，pitch方向的旋转范围为顺时针5°到30°，roll方向的旋转范围为顺时针5°到30°。

# 3.Results

对所有测试数据，三个定量指标的均值、标准差以及最大误差见表1：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/5.png)

$\alpha$和$d$的分布见Fig3。可以看到，偏差值大多集中在0附近。75%的数据满足$\alpha < 1 °$和$d < 1 mm$。说明检测到的MSP和GT差距不大。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/6.png)

此外，为了验证$\alpha$和$\theta_{GT}$以及$d$和$\theta_{GT}$之间是否存在相关性，作者还进行了两种相关性检验，结果见表2。其中，Kendall's $\tau$为非参数检验，Pearson线性相关为参数检验。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/7.png)

对于旋转数据的测试结果见表3：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/8.png)

实验表明，为了保证检测MSP的准确性，yaw的最大旋转角度为40°，pitch的最大旋转角度为30°，roll的最大旋转角度为25°。

$d$随旋转角度的变化而变化的趋势见Fig4。实线表示yaw方向，点线表示pitch方向，虚线表示roll方向。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/9.png)

# 4.Discussion

## 4.1.Parameter justification

### 4.1.1.Selection of threshold values for $\theta_d$ and $lr\\_ratio$

这部分主要用于解释为什么使用$\theta_d$和$lr\\_ratio$来判断原始数据是否需要旋转。首先，MSP的检测对pitch方向的旋转并不敏感。$\theta_d$用于衡量yaw方向的角度偏差。$lr\\_ratio$用于衡量roll方向的角度偏差。而7.8°和10%这两个阈值则是通过之前提到的8折交叉验证得到的。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/10.png)

Fig5是在寻找最优阈值时做的一些实验，此处不再详述。

### 4.1.2.Selection of KL search range

在[论文“Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure”](https://shichaoxin.com/2026/08/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Extraction-of-the-midsagittal-plane-from-morphological-neuroimages-using-the-Kullback-Leibler-s-measure/#23algorithm)中设置了20mm的KL搜索范围，这里对从5mm到40mm的搜索范围进行了一个评估，结果见Fig6。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/11.png)

### 4.1.3.Selection of reference axial slice for ellipse fitting

这部分主要解释$slice\\_ref$的确定依据，不再详述。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/12.png)

## 4.2.Accuracy, robustness and speed

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/13.png)

[21]就是[论文“Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure”](https://shichaoxin.com/2026/08/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Extraction-of-the-midsagittal-plane-from-morphological-neuroimages-using-the-Kullback-Leibler-s-measure/)提出的方法。

Fig10是在低分辨率CT数据上的结果：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/14.png)

Fig11是$\alpha$和$\theta_{GT}$之间的关系：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/15.png)

Fig12是$d$和$\theta_{GT}$之间的关系：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/16.png)

Fig13是一个IF弯曲的案例：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/17.png)

在3 GHz CPU（1 GB RAM）硬件条件下，24层的CT数据，算法运行时间在10秒左右。

## 4.3.Comparison with other techniques

### 4.3.1.Comparison with cross-correlation technique for MSP extraction

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/18.png)

### 4.3.2.Comparison with principal axis method

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP/19.png)

# 5.Conclusion

该算法已被集成到作者开发的急诊卒中CAD系统中。

# 6.论文链接

👽[Robust calculation of the midsagittal plane in CT scans using the Kullback–Leibler’s measure](https://github.com/x-jeff/AI_Papers/blob/master/2026/Robust%20calculation%20of%20the%20midsagittal%20plane%20in%20CT%20scans%20using%20the%20Kullback%E2%80%93Leibler%E2%80%99s%20measure.pdf)