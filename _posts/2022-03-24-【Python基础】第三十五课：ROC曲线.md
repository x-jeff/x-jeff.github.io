---
layout:     post
title:      【Python基础】第三十五课：ROC曲线
subtitle:   LabelEncoder，predict_proba，roc_curve，auc
date:       2022-03-24
author:     x-jeff
header-img: blogimg/20220324.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.使用Python计算ROC曲线

ROC和AUC的相关介绍见：[ROC与AUC](http://shichaoxin.com/2018/12/03/机器学习基础-第三课-模型性能度量/#4roc与auc)。

👉载入必要的包：

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
```

👉数据读取与编码转换：

```python
iris = load_iris()
X = iris.data[50:150, ]

le = preprocessing.LabelEncoder()
y = le.fit_transform(iris.target[50:150])
```

`LabelEncoder`将类别标签进行编码（0～类别-1）：

```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_ # output : array([1, 2, 6])
le.transform([1, 1, 2, 6]) # output : array([0, 0, 1, 2])
le.inverse_transform([0, 0, 1, 2]) # output : array([1, 1, 2, 6])
```

也可以应用于字符串：

```python
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
list(le.classes_) # output : ['amsterdam', 'paris', 'tokyo']
le.transform(["tokyo", "tokyo", "paris"]) # output : array([2, 2, 1])
list(le.inverse_transform([2, 2, 1])) # output : ['tokyo', 'tokyo', 'paris']
```

* `le.fit`：Fit label encoder。
* `le.fit_transform`：Fit label encoder and return encoded labels。

👉建立预测模型：

```python
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=123)
clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)
```

`train_test_split`用法见：[【Python基础】第三十四课：模型评估方法](http://shichaoxin.com/2022/02/15/Python基础-第三十四课-模型评估方法/)。

`DecisionTreeClassifier()`用法见：[【Python基础】第二十七课：分类模型之决策树](http://shichaoxin.com/2021/10/17/Python基础-第二十七课-分类模型之决策树/)。

👉计算ROC Curve参数：

```python
probas_ = clf.fit(train_X, train_y).predict_proba(test_X)
probas_[:, 1]

fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])
```

`predict_proba`返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个样本为第j个类别的概率，每一行的概率和为1。

`roc_curve`返回的`fpr`、`tpr`、`thresholds`均为数组。

👉绘制ROC Curve：

```python
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
plt.show()
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson35/35x1.png)

👉计算AUC分数：

```python
from sklearn.metrics import auc

roc_auc = auc(fpr, tpr) # roc_auc=0.876838
```

# 2.代码地址

1. [ROC曲线](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo35)

# 3.参考资料

1. [sklearn中predict_proba用法（注意和predict的区别）](https://blog.csdn.net/u011630575/article/details/79429757)