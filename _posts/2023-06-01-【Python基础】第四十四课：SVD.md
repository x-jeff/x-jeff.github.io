---
layout:     post
title:      【Python基础】第四十四课：SVD
subtitle:   scipy.linalg.svd，sklearn.decomposition.TruncatedSVD
date:       2023-06-01
author:     x-jeff
header-img: blogimg/20220509.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.SVD

请见：[【数学基础】第十七课：奇异值分解](http://shichaoxin.com/2020/11/24/数学基础-第十七课-奇异值分解/)。

# 2.使用SVD做矩阵还原

使用[【数学基础】第十七课：奇异值分解](http://shichaoxin.com/2020/11/24/数学基础-第十七课-奇异值分解/)中的例子：

$$B=PDQ^T=\begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 0 \\ 0 & 0 \\ \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \end{bmatrix}^T = \begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 0 & 0 \\ \end{bmatrix}$$

使用代码实现：

```python
import numpy as np
from scipy.linalg import svd

X = np.array([[1,1],
              [1,1],
              [0,0]])
             
U,S,V = svd(X, full_matrices=True)
```

`full_matrices`是一个布尔值，指定是否返回完整的奇异值矩阵。默认为True，表示返回完整的奇异值矩阵；如果设为False，则返回一个经过[截断的奇异值矩阵](http://shichaoxin.com/2022/03/07/论文阅读-Fast-R-CNN/#31truncated-svd-for-faster-detection)。

根据上述代码，我们得到的U为：

$$U = \begin{bmatrix} -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \\ \end{bmatrix} $$

相比之前的P，这里将第一列的特征向量乘了个-1，并不影响结果成立（详见：[矩阵的特征值和特征向量](http://shichaoxin.com/2020/08/12/数学基础-第十五课-矩阵的相似变换和相合变换/#121矩阵的特征值和特征向量)）。

这里的S只输出了对角线上的奇异值：

$$S = \begin{bmatrix} 2 & 0 \\ \end{bmatrix} $$

输出的V为：

$$V = \begin{bmatrix} -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \end{bmatrix}$$

这里的V是把Q中每一列的特征向量都乘了-1并转置之后得到的。

如果我们将`full_matrices`设为False，得到的结果如下：

$$\begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 0 & 0 \\ \end{bmatrix} = \begin{bmatrix} -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & 0\\ \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 0 \\ \end{bmatrix} \begin{bmatrix} -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \end{bmatrix}$$

如果我们使用另一个直接使用[截断的奇异值矩阵](http://shichaoxin.com/2022/03/07/论文阅读-Fast-R-CNN/#31truncated-svd-for-faster-detection)的API：

```python
from sklearn.decomposition import TruncatedSVD
X = np.array([[1,1],
              [1,1],
              [0,0]])
svd = TruncatedSVD(1)
x = svd.fit_transform(X)
```

TruncatedSVD(1)中的1是n\_components，即输出数据的期望维数，也就是[Truncated SVD](http://shichaoxin.com/2022/03/07/论文阅读-Fast-R-CNN/#31truncated-svd-for-faster-detection)中t的值。先对$X$进行[Truncated SVD](http://shichaoxin.com/2022/03/07/论文阅读-Fast-R-CNN/#31truncated-svd-for-faster-detection)分解（t=1）：

$$\begin{bmatrix} 1 & 1 \\ 1 & 1 \\ 0 & 0 \\ \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \\ 0 \\ \end{bmatrix} \begin{bmatrix} 2  \\ \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \end{bmatrix}$$

则降维后的$x$其实就是：

$$\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \\ 0 \\ \end{bmatrix} \begin{bmatrix} 2  \\ \end{bmatrix} = \begin{bmatrix} \sqrt{2} \\ \sqrt{2} \\ 0 \\ \end{bmatrix}$$

# 3.代码地址

1. [SVD](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo44)