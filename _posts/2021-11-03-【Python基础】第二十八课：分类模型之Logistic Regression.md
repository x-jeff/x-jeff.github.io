---
layout:     post
title:      【Python基础】第二十八课：分类模型之Logistic Regression
subtitle:   LogisticRegression()
date:       2021-11-03
author:     x-jeff
header-img: blogimg/20211103.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.建立逻辑回归分析模型

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
clf = LogisticRegression()
clf.fit(iris.data, iris.target)

clf.predict(iris.data)
```

和上篇博客[【Python基础】第二十七课：分类模型之决策树](http://shichaoxin.com/2021/10/17/Python基础-第二十七课-分类模型之决策树/)中所做的一样，我们也可以绘制其决策边界：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson28/28x1.png)

# 2.代码地址

1. [LogisticRegression](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo28)