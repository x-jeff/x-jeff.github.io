---
layout:     post
title:      【Python基础】第二十五课：SQL实战应用之汇率资讯储存与管理
subtitle:   SQL实战应用之汇率资讯储存与管理
date:       2021-08-23
author:     x-jeff
header-img: blogimg/20210823.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.数据收集

通过网络爬虫收集汇率信息（网络爬虫的实现参照博客：[【Python基础】第八课：网络爬虫
](http://shichaoxin.com/2019/11/04/Python基础-第八课-网络爬虫/)）。所爬取的汇率信息来自[国家外汇管理局官方网站](http://www.safe.gov.cn)：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson25/25x1.png)

确定目标内容的位置以及HTTP请求方法：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson25/25x2.png)

>HTTP请求可以有很多方法。HTTP1.0定义了三种请求方法：`GET`、`POST`和`HEAD`。HTTP1.1新增了五种请求方法：`OPTIONS`、`PUT`、`DELETE`、`TRACE`和`CONNECT`。在此不再区分这几种方法的异同。

确定限定日期所用的字段名称：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson25/25x3.png)

```python
import requests

payload = {'startDate': '2021-05-01', 'endDate': '2021-08-01', 'queryYN': 'true'}
res = requests.post('http://www.safe.gov.cn/AppStructured/hlw/RMBQuery.do', data=payload)
print(res)
```

输出为：

```python
<Response [200]> #证明网页响应成功
```

tips：该网页在输入日期范围时，跨度不能超过366天。此时如果需要多年的数据，可以使用[`datetime`](http://shichaoxin.com/2020/08/19/Python基础-第十五课-处理时间格式资料/)生成一系列的日期，然后循环调用：

```python
from datetime import datetime, timedelta

current_time = datetime.now()
for i in range(1, 5 * 366, 366):
    start_time = (current_time - timedelta(days=i + 366)).strftime('%Y%m%d')
    end_time = (current_time - timedelta(days=i + 1)).strftime('%Y%m%d')
    print(start_time, end_time)
```

输出为：

```
20200822 20210822
20190822 20200821
20180821 20190821
20170820 20180820
20160819 20170819
```

使用`pandas.read_html`快速获取在html页面中table格式的数据：

```python
from bs4 import BeautifulSoup
import pandas as pd

soup = BeautifulSoup(res.text, 'html.parser')
dfs = pd.read_html(soup.select('#InfoTable')[0].prettify('utf-8-sig'), header=0)
df_rates = dfs[0]
print(df_rates.head())
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson25/25x4.png)

# 2.数据转换

```python
df_rates = pd.melt(df_rates, col_level=0, id_vars=['日期'])
df_rates.columns = ['date', 'currency', 'exchange']
print(df_rates.head())
```

输出为：

```
         date currency  exchange
0  2021-07-30     美元    646.02
1  2021-07-29     美元    649.42
2  2021-07-28     美元    649.29
3  2021-07-27     美元    647.34
4  2021-07-26     美元    647.63
```

>`pandas.melt`的用法：[链接](http://shichaoxin.com/2021/07/30/Python基础-第二十四课-SQL-Query的使用/#21pandasmelt)。

# 3.将数据储存到数据库中

```python
import sqlite3 as lite

with lite.connect('currency.sqlite') as db:
    df_rates.to_sql('currency_data', con=db, if_exists='replace', index=None)
```

选取美元汇率数据：

```python
with lite.connect('currency.sqlite') as db:
    df = pd.read_sql("SELECT * FROM currency_data WHERE currency='美元'", con=db)
    print(df.head())
```

输出为：

```
         date currency  exchange
0  2021-07-30     美元    646.02
1  2021-07-29     美元    649.42
2  2021-07-28     美元    649.29
3  2021-07-27     美元    647.34
4  2021-07-26     美元    647.63
```

绘制美元和英镑的汇率折线图：

```python
with lite.connect('currency.sqlite') as db:
    df = pd.read_sql("SELECT * FROM currency_data WHERE currency IN ('美元','英镑')", con=db)
    df.currency.unique() #去除重复
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df2 = df.pivot_table(index='date', columns='currency')
    df2.plot(kind='line', rot=90)
    plt.show()
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson25/25x5.png)

>使用pandas绘制统计图表请参见：[【Python基础】第二十课：使用pandas绘制统计图表](http://shichaoxin.com/2021/05/14/Python基础-第二十课-使用pandas绘制统计图表/)。
>
>`pivot_table`的使用说明请参见：[建立透视表（pivot_table）](http://shichaoxin.com/2020/09/25/Python基础-第十六课-重塑资料/#2建立透视表pivot_table)。

# 4.代码地址

1. [SQL实战应用之汇率资讯储存与管理](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo25)