---
layout:     post
title:      【Python基础】第二十课：使用pandas绘制统计图表
subtitle:   pandas.DataFrame.plot()，pandas.DataFrame.rolling()
date:       2021-05-14
author:     x-jeff
header-img: blogimg/20210514.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.读取数据

我们以爬取的股价信息为例：

```python
import pandas_datareader
df = pandas_datareader.DataReader('BABA', data_source='yahoo', start='2020-05-01')
print(df.head())
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson20/20x1.png)

# 2.`pandas.DataFrame.plot()`

pandas画图函数的API为：

```python
DataFrame.plot(x=None, y=None, kind='line', ax=None, subplots=False, 
                sharex=None, sharey=False, layout=None, figsize=None, 
                use_index=True, title=None, grid=None, legend=True, 
                style=None, logx=False, logy=False, loglog=False, 
                xticks=None, yticks=None, xlim=None, ylim=None, rot=None, 
                fontsize=None, colormap=None, position=0.5, table=False, yerr=None, 
                xerr=None, stacked=True/False, sort_columns=False, 
                secondary_y=False, mark_right=True, **kwds)
```

部分参数解释：

1. `x`：x轴label。
2. `y`：y轴label。
3. `kind`：图类型：
	* `"line"`：折线图（默认类型）。
	* `"bar"`：条形图。
	* `"barh"`：横向条形图。
	* `"hist"`：直方图。
	* `"box"`：箱形图。
	* `"kde"`：Kernel Density Estimation plot（[wiki](https://en.wikipedia.org/wiki/Kernel_density_estimation)）。
	* `"density"`：same as "kde"。
	* `"area"`：area plot（[wiki](https://en.wikipedia.org/wiki/Area_chart)）。
	* `"pie"`：饼图。
	* `"scatter"`：散点图。
	* `"hexbin"`：六边形图（hexbin plot）。
4. `figsize`：图的大小，用一个tuple表示：(width,height)（单位为英寸）。
5. `title`：图的名字或标题。
6. `grid`：背景是否添加网格线。
7. `legend`：是否显示图例。

## 2.1.绘制折线图

```python
df['Close'].plot(kind='line', figsize=[10, 5], title='BABA', legend=True, grid=True)
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson20/20x2.png)

## 2.2.绘制移动平均线

```python
df['mvg30'] = df['Close'].rolling(window=30).mean()
df[['Close', 'mvg30']].plot(kind='line', legend=True, figsize=[10, 5])
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson20/20x3.png)

### 2.2.1.`pandas.DataFrame.rolling()`

在建模过程中，我们常常需要对有时间关系的数据进行整理。比如我们想要得到某一时刻过去30分钟的销量（产量，速度，消耗量等），传统方法复杂消耗资源较多，pandas提供的rolling使用简单，速度较快。

```python
DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)
```

参数解释：

1. `window`：表示时间窗的大小，注意有两种形式（int or offset）。如果使用int，则数值表示计算统计量的观测值的数量，即向前几个数据。如果是offset类型，则表示时间窗的大小。
2. `min_periods`：最少需要有值的观测点的数量，对于int类型，默认与window相等。对于offset类型，默认为1。
3. `center`：是否使用window的中间值作为label，默认为false。只能在window是int时使用。
4. `win_type`：窗口类型。默认为None，一般不特殊指定。
5. `on`：对于DataFrame如果不使用index（索引）作为rolling的列，那么用on来指定使用哪列。
6. `axis`：方向（轴），一般都是0。
7. `closed`：定义区间的开闭，曾经支持int类型的window，新版本已经不支持了。对于offset类型默认是左开右闭的，即默认为right。可以根据情况指定为left,both等。

## 2.3.绘制直方图

```python
df.loc[df.index >= '2021-05-01', 'Volume'].plot(x='datetime', kind='bar', figsize=[10, 5], title='BABA', legend=True)
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson20/20x4.png)

## 2.4.饼图

```python
df['diff'] = df['Close'] - df['Open']
df['rise'] = df['diff'] > 0
df['fall'] = df['diff'] < 0
df[['rise', 'fall']].sum().plot(kind='pie', figsize=[5, 5], counterclock=True, startangle=90, legend=True)
```

![](https://github.com/x-jeff/BlogImage/raw/master/PythonSeries/Lesson20/20x5.png)

# 3.代码地址

1. [使用pandas绘制统计图表](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo20)

# 4.参考资料

1. [【python】详解pandas.DataFrame.plot( )画图函数](https://blog.csdn.net/brucewong0516/article/details/80524442)
2. [pandas.DataFrame.plot](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)
3. [pandas.DataFrame.rolling](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
4. [pandas中时间窗函数rolling的使用](https://blog.csdn.net/wj1066/article/details/78853717)