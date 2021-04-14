---
layout:     post
title:      【Python基础】第十九课：使用pandas产生叙述性统计
subtitle:   叙述性统计，pandas_datareader，pct_change
date:       2021-04-14
author:     x-jeff
header-img: blogimg/20210414.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.叙述性统计与推论性统计

**叙述性统计：**

* 有系统的归纳数据，了解数据的轮廓。
* 对数据样本做叙述性陈述，例如：平均数、标准偏差、计次频率、百分比。
* 对数据资料的图像化处理，将数据摘要变为图表。

**推论性统计：**

* 资料模型的构建。
* 从样本推论整体资料的概况。
* 相关、回归、单因子变异数、因素分析。

# 2.使用pandas产生叙述性统计

pandas提供了一个专门从财经网站获取金融数据的API接口，封装在`pandas_datareader`中。我们用这个API来获取用于统计的数据：

```python
import pandas_datareader

df = pandas_datareader.data.DataReader('BABA', data_source='yahoo')
print(df.tail())
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x1.png)

>`pandas_datareader`的用法见本文第3部分。

## 2.1.简易统计

```python
# 算出总和
df['Close'].sum()
# 算出平均
df['Close'].mean()
# 算出标准差
df['Close'].std()
# 取得最小值
df['Close'].min()
df[['Open', 'Close']].min()
# 取得最大值
df['Close'].max()
df[['Open', 'Close']].max()
# 取得笔数
df['Close'].count()
```

```python
# 取得整体叙述性统计
df.describe()
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x2.png)

## 2.2.基本计算

```python
# 计算当日涨跌
df['diff'] = df['Close'] - df['Open']
df['rise'] = df['diff'] > 0
df['fall'] = df['diff'] < 0
# 计算涨跌次数
df[['rise', 'fall']].sum()
# 计算当月涨跌次数
df.loc[df.index >= '2017-04-01', ['rise', 'fall']].sum()#这里只能用.loc，不能用.iloc
```

`df.index`：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x3.png)

>index的格式为[datetime](http://shichaoxin.com/2020/08/19/Python基础-第十五课-处理时间格式资料/)。

```python
# 根据年月统计涨跌次数
df.groupby([df.index.year, df.index.month])['rise', 'fall'].sum()
```

>groupby的用法见[链接](http://shichaoxin.com/2020/02/23/Python基础-第十一课-处理缺失值/#3groupby和transform)。

```python
# 计算每日报酬
df['ret'] = df['Close'].pct_change(1)
```

>`pct_change`的用法见本文第4部分。

# 3.`pandas_datareader.data.DataReader`

```python
def DataReader(
    name,
    data_source=None,
    start=None,
    end=None,
    retry_count=3,
    pause=0.1,
    session=None,
    api_key=None,
)
```

参数讲解：

1. `name`：股票名称。
2. `data_source`：数据来源。
3. `start`：起始时间。
4. `end`：终止时间。
5. `retry_count`：如果获取数据失败，则尝试重新获取的次数。
6. `pause`：尝试重新获取数据的时间间隔。
7. `session`：requests.sessions.Session instance to be used。
8. `api_key`：specify an API key for certain data sources。

# 4.`pct_change`

pandas中DataFrame的`pct_change`用于计算当前元素与先前元素相差的百分比。

```python
def pct_change(self, periods=1,fill_method='pad', limit=None, freq=None,**kwargs)
```

1. `periods=n`：表示当前元素与先前第n个元素相差的百分比。
2. `fill_method`：缺失值的填补方式，默认为[pad方式](http://shichaoxin.com/2020/02/23/Python基础-第十一课-处理缺失值/#2214向前后填值)。
3. `limit`：填补连续缺失值的个数限制。
4. `freq`：DateOffset, timedelta, or offset alias string, optional. Increment to use from time series API (e.g. 'M' or BDay()).
5. `**kwargs`：Additional keyword arguments are passed into `DataFrame.shift` or `Series.shift`.

👉例子一：

```python
s = pd.Series([90, 91, 85])
```

`s.pct_change()`为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x4.png)

$$\frac{91-90}{90}=0.011111;\frac{85-91}{91}=-0.065934$$

`s.pct_change(periods=2)`为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x5.png)

$$\frac{85-90}{90}=-0.055556$$

👉例子二：

```python
s = pd.Series([90, 91, None, 85])
```

`s.pct_change(fill_method='ffill')`为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x6.png)

👉例子三：

```python
df = pd.DataFrame({
             'FR': [4.0405, 4.0963, 4.3149],
             'GR': [1.7246, 1.7482, 1.8519],
             'IT': [804.74, 810.01, 860.13]},
             index=['1980-01-01', '1980-02-01', '1980-03-01'])
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x7.png)

`df.pct_change()`为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x8.png)

`df.pct_change(axis='columns')`为：

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson19/19x9.png)

# 5.代码地址

1. [使用pandas产生叙述性统计](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo19)