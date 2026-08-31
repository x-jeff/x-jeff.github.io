---
layout:     post
title:      【论文阅读】Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure
subtitle:   Midsagittal plane，Interhemispheric fissure，Kullback–Leibler’s measure，Talairach transformation
date:       2026-08-26
author:     x-jeff
header-img: blogimg/20210810.jpg
catalog: true
tags:
    - Medical Imaging
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

表1列出了之前MSP（MidSagittal Plane）提取的一些方法：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/1.png)

本文提出使用[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)直接从三维体数据中提取MSP。该算法平均耗时约为5秒，且可应用于CT和MRI数据。

# 2.Method

## 2.1.Definitions

这部分主要讲了[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)，这里不再详述。

假设有两个离散概率分布：$p = \\{ p_i \\}$和$q = \\{ q_i \\}$。其中，$p_i$表示事件$i$在分布$p$中的发生概率，$q_i$表示事件$i$在分布$q$中的发生概率。两个离散概率分布$p$和$q$的[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)可计算为：

$$I(p/q) = \sum_i p_i \log p_i - \sum_i p_i \log q_i = \sum_i p_i \log (p_i / q_i) \tag{1}$$

式(1)可改写为：

$$I(p/q) = H(p) - K(p,q) \tag{2}$$

其中，$H(p)=\sum_i p_i \log p_i$是Shannon熵的负值，$K(p,q)=\sum_i p_i \log q_i$通常被称为Kerridge不准确度。

## 2.2.Problem statement

让$p = \\{ p_i \\}$和$q = \\{ q_i \\}$分别表示两个sagittal平面的灰度值离散概率分布。$p_i$和$q_i$分别表示灰度值$i$在分布$p$和分布$q$出现的概率。这些概率值由图像的灰度直方图计算得到。在实际计算时，如果出现$p_i = 0$，则定义$0 \log (0 / q_i) = 0$；如果出现$q_i = 0$，则忽略$p_i \log (p_i / 0)$。

MSP可定义为一个穿过大脑半球间裂（the Interhemispheric Fissure，IF）的平面，并且以该平面为中心，一些脑部结构呈现出近似的双侧对称性。在正常的数据中，MSP也可认为是这样的一个平面：在所有矢状位切片中，它包含的脑脊液（cerebrospinal fluid，CSF）数量最多。理想情况下，如果满足以下两个条件：

* IF没有发生弯曲。
* 体素在左右方向上的尺寸小于IF的宽度（这能保证一个体素能完全落在IF内，不会横跨左右半球，避免部分容积效应）。

那么MSP通常包含以下脑部结构：

* 前连合（Anterior Commissure，AC）和后连合（Posterior Commissure，PC）。
* 第三脑室、第四脑室、中脑导水管以及IF中的脑脊液。
* 一些结构的横断面，比如胼胝体、脑干、松果体、小脑、丘脑间粘合（如果存在）等。
* 不包含大脑皮层区域的横断面。

这些解剖结构特征使得MSP能够与其他平面区分开来，并且这种差异会清楚地反映在灰度直方图中。对于一个典型的T1加权三维图像（MR图像），MSP的灰度直方图有以下2个特点：

* 在脑脊液对应的灰度范围处有一个明显的高峰。
* 在脑白质对应的灰度范围处有一个低峰。

而对于一张远离MSP的矢状位切片，其灰度直方图则相反：

* 脑脊液对应位置的峰较低。
* 脑白质对应位置的峰较高。

如Fig1所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/2.png)

在Fig1中，横轴表示灰度值，纵轴表示对应灰度值的像素数量。数据来自模体。红色虚线表示MSP，蓝色实线表示距离MSP 2cm的另一张矢状位切片。峰值A是背景。

[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)是一种相对度量，在计算过程中，它会比较两张切片，对于一些影响因素，比如图像噪声、图像不均匀性等，这种比较在一定程度上起到了归一化的作用。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/3.png)

在Fig2中，横轴为不同的矢状位切片，纵轴为归一化后的[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)或Shannon熵。虚线表示归一化[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)，实线表示归一化Shannon熵。[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)需要两张切片对比才能计算得到，而Shannon熵只需要一张切片即可计算。从Fig2可以看出，虚线的峰值不一定刚好对应实线的谷底，所以二者并不存在严格的对应关系。

我们评估了多种函数或度量方法，最终得到结论，[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)能够给出MSP最好的近似估计。

## 2.3.Algorithm

算法分为两个阶段：

* 第一个阶段：通过分析矢状位切片，计算得到一个粗略近似的MSP，记为cMSP（coarse MSP）。
* 第二个阶段：在cMSP的基础上进一步细化，从而得到真实MSP的最佳近似。

用$(v_x,v_y,v_z)$表示体素在三个方向上的物理尺寸，单位为mm。假设体素坐标均为整数。在三维数据场中，X轴对应右左方向，Y轴对应前后方向，Z轴对应上下方向。

算法第一阶段的步骤如下：

* 读取脑部三维数据、体素尺寸以及图像方向信息。初始volume可以是各向同性，也可以是各向异性。
* 如果原始数据的方向是轴位或冠状位，则生成对应的矢状位切片。
* 确定整个volume的矢状位中央切片。
* 以中央切片为中心，取±2cm范围内的矢状位切片作为VOI。
* 选择VOI中的第一张切片作为参考切片。
* 对VOI中的所有切片，分别计算它们相对于参考切片的[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)。
* 选择[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)最大的那张切片作为cMSP，如Fig3所示。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/4.png)

设$s$为cMSP对应的矢状位切片索引（即X坐标）。cMSP的4个角点可分别定义为：$(s,y_1,z_1),(s,y_2,z_2),(s,y_3,z_3),(s,y_4,z_4)$（使用像素/体素坐标，不是毫米物理坐标）。算法第二阶段的步骤如下：

1. 因为3个不共线的点可以确定一个平面，因此第一步先选择cMSP的任意3个角点，比如$(a_0,y_1,z_1),(b_0,y_2,z_2),(c_0,y_3,z_3)$，且$s=a_0=b_0=c_0$。
2. 沿X轴方向，距离cMSP 2cm位置处定义一个参考矢状位切片，在整个第二阶段，这张参考切片保持不变。
3. 设置初始VOI为第一阶段定义的VOI，$L$为VOI内矢状位切片的数量。初始步长$\Delta_0$定义为：$\Delta _0 = \min (\lceil L / 8 \rceil, 8)$。扰动的搜索范围需限定在VOI内。
4. 对$(a_0,y_1,z_1),(b_0,y_2,z_2),(c_0,y_3,z_3)$进行扰动，扰动范围为$a_0,b_0,c_0 \in [s \pm \Delta_0]$，即$a_0,b_0,c_0$均有3个取值，分别为$s - \Delta_0, s, s + \Delta_0$。也就是说，这一步一共产生了27个候选平面。
5. 计算所有候选平面与参考切片之间的[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)，得到[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)最大的那个平面，其角点可记为$(a_k,y_1,z_1),(b_k,y_2,z_2),(c_k,y_3,z_3)$，$k$表示迭代次数。步长更新为$\Delta_k = \lceil \frac{\Delta_{k-1}}{2} \rceil$，VOI在X方向的长度更新为$4 \Delta_k$。接下来继续扰动，以第一个角点为例，$a_k$的扰动取值范围为$[a_k \pm 2\Delta_k]$，也就是取值可以是$a_k - 2\Delta_k, a_k - \Delta_k, a_k, a_k + \Delta_k, a_k + 2\Delta_k$。相当于这一步一共产生了125个候选平面，计算每个候选平面与参考切片之间的[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)。
6. 只要步长$\Delta_k$大于等于1个像素，就重复执行第5步。
7. 最终得到的使[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)达到最大的那个平面，就被认为是MSP的最佳近似平面。

第一阶段和第二阶段得到的结果见Fig4：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/5.png)

## 2.4.Data overview

研究所用数据来自日本长崎大学医学院（Nagasaki University School of Medicine，NUSM）、新加坡中央医院（Singapore General Hospital，SGH）、互联网脑分割数据库（Internet Brain Segmentation Repository，IBSR）以及BrainWeb的模体数据。

研究使用了多模态数据集，包括MRI、MRA和CT，见表2：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/6.png)

MR数据采用多种脉冲序列采集，包括：T1WI（T1 weighted image）、SPGR（spoiled gradient recovery）、contrast enhanced SPGR、FLAIR（fluid attenuated inversion recovery）、T2WI（T2 weighted image）以及PD（proton density）。这些数据在进行算法处理前没有经过任何预处理。图像的层厚范围为1到8mm，层数范围为8到320层。部分数据存在较明显的部分容积效应和严重的强度不均匀性。

## 2.5.Generation of the ground truth lines

对三维数据的每一层，即轴位2D图像，神经解剖学专家都会在上面绘制IF的真实标注线（Ground Truth Lines，GTL）。为了定量评估算法结果，使用了以下3个指标：

1. 角度偏差$\alpha$：算法计算得到的MSP和各条GTL之间夹角的平均值。
2. 距离偏差$d$：各条GTL的端点到MSP的欧氏距离的平均值。这里的距离偏差以像素为单位表示。GTL为线段，所以有两个端点。距离偏差会受到标注的GTL端点位置的影响。
3. 平面性偏差$\alpha_{GTL}$：所有GTL之间的最大角度偏差。这个指标用来衡量IF本身的非平面程度。

## 2.6.Validation of results

如果IF本身是弯曲的，无法近似成一个平面，那也就不能期待计算得到的MSP可以很好的拟合所有的GTL。于是作者使用了Kendall's $\tau$相关性检验和Pearson线性相关性检验，目的是验证随着IF非平面程度的加重，MSP和GTL之间的误差会越来越大，从而区分算法自身的误差和IF非平面程度导致的误差。

此外，作者还使用所有GTL通过最小二乘误差（least square error，LSE）拟合了一个平面，并将计算的MSP和LSE拟合平面进行了定性比较。

# 3.Results

Fig5-8展示了在不同数据集上得到的MSP投影结果。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/7.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/8.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/9.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/10.png)

计算的MSP和LSE拟合平面的比较见Fig9：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/11.png)

$\alpha, d$与$\alpha_{GTL}$之间的相关性检验结果见表3：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/12.png)

从表3可以看出，$\alpha$与$\alpha_{GTL}$之间存在显著相关性，也就是说，IF非平面程度越严重，算法得到的MSP与GTL之间的角度误差往往也越大。但$d$与$\alpha_{GTL}$之间没有显著相关性。

多模态数据的结果见表4：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/13.png)

为了分析噪声、强度不均匀以及层厚对算法的影响，作者使用了BrainWeb数据集对算法进行了测试，结果见表5：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/14.png)

在表5中，“Dataset”列的4位编号的含义为：第一位数字表示层厚；第二位数字表示噪声百分比；后两位数字表示RF强度不均匀性百分比。例如编号“1040”表示层厚为1mm，噪声百分比为0%，RF强度不均匀性百分比为40%。当层厚为1mm时，$\alpha_{GTL}$为1.78°；当层厚为3mm时，$\alpha_{GTL}$为1.45°。

作者还使用了编号为“1940”的数据来测试算法对旋转的鲁棒性，在pitch、yaw、roll三个方向上对数据进行旋转。数据分别进行了顺时针和逆时针旋转，pitch和yaw方向的旋转角度范围为1°到15°，roll方向的旋转角度范围为1°到10°。结果表明，该算法在一定范围内对旋转具有较好的鲁棒性。当yaw、pitch、roll分别单独变化时，算法可在-15°\~15°范围内保持较好的鲁棒性。当roll与其他旋转方向组合时，roll的鲁棒范围约为-7°\~7°。顺时针和逆时针旋转得到的结果大致相同。Fig10和Fig11展示了逆时针旋转的结果。注意，旋转后的体数据是通过三线性插值构建的。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/15.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/MedicalImaging/MSP1/16.png)

算法的执行时间取决于VOI中体素的数量。作者所采用的coarse-to-fine搜索算法的计算复杂度为$O(n)$，其中$n$表示初始VOI中的矢状位切片数量。算法使用MS VC++实现。例如，对于一个尺寸为$256 \times 256 \times 168$的数据，体素尺寸为$1.0 \times 1.1 \times 0.66$，在Windows XP、Pentium 4, 2.4 GHz、512 MB RAM的硬件环境中，算法执行时间不到5秒。

# 4.Discussion

不同个体之间的大脑存在差异，此外，同一脑组织在不同模态、不同扫描机器、不同扫描协议下的图像也可能都不一样。因此很难把整个图像归一化到某个固定标准，因此，作者采用的方法是在数据中选取一张参考切片，将其他切片相对于这张参考切片进行归一化，分析各切片相对于参考切片的相对变化。

## 4.1.Kullback–Leibler’s measure and entropy

与单纯使用熵度量相比，[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)能够更准确地近似得到cMSP和最终MSP。

## 4.2.Assumptions

假定roll方向的旋转角度不超过7°。

## 4.3.Justification of parameters and selection of the reference slice

用于选择参考切片的参数，即距离初始选定切片20mm，虽然是通过经验确定的，但它具有一定的解剖学依据。作者还测试了不同的参考切片距离，范围从5mm到40mm，实验表明，20mm的结果是最好的。

在之前的算法描述中，我们只选择了一张参考切片，这适用于正常病例。如果图像中存在病变区域，而参考切片恰好穿过该病变区域，那么计算得到的MSP就可能不准确。因此，后来作者优化了算法，使用左右半球各一张参考切片。在算法的第一阶段，对于每一个候选切片，分别计算其与左参考切片和右参考切片的[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)，然后将这两个[KL度量](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)相加，并将相加后的值归一化到0-1，归一化后值最大的候选切片，会被认为是cMSP的位置。类似地，作者在算法的第二阶段也使用了2个参考切片。

算法第二阶段设置的初始步长$\Delta_0$也是通过经验选择的，作者在不同数据集上尝试了多种不同的初始步长，最终选用了描述中的设置，因为它能实现更快的收敛速度。

## 4.4.Noise, RF-inhomogeneity, and rotation

噪声、RF强度不均匀性以及部分容积效应并没有影响算法的准确性（见表5）。这是因为参考切片和MSP中包含的噪声和强度不均匀性水平大致相同，因此这些影响在相对比较时可以相互抵消。

旋转范围的限制见第3部分。

## 4.5.Ground truth lines and MSP

由于GTL只在IF可见的位置进行标注，因此不同切片上的GTL长度可能不同，所以算法得到的MSP与GTL之间的距离偏差$d$，不仅取决于$\alpha_{GTL}$，也受到GTL线段长度的影响。这也解释了为什么$d$和$\alpha_{GTL}$之间没有显著相关性。

## 4.6.MSP versus LSE

从Fig9可以看出，MSP与LSE拟合平面是不同的，从可见的解剖结构来说，MSP的定义更好。

## 4.7.Applicability of the algorithm to different modalities, pulse sequences, and pathological datasets

该算法适用于不同模态的数据。并且不但适用于正常病例，也适用于病理病例的数据。

# 5.Conclusion

不再赘述。

# 6.论文链接

👽[Extraction of the midsagittal plane from morphological neuroimages using the Kullback–Leibler’s measure](https://github.com/x-jeff/AI_Papers/blob/master/2026/Extraction%20of%20the%20midsagittal%20plane%20from%20morphological.pdf)