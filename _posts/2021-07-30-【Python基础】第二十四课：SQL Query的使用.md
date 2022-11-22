---
layout:     post
title:      【Python基础】第二十四课：SQL Query的使用
subtitle:   pandas.melt，pandas.read_csv，pandas.to_sql，pandas.read_sql，SELECT，FROM，WHERE，ORDER BY，DESC，LIMIT，AVG，GROUP BY，HAVING
date:       2021-07-30
author:     x-jeff
header-img: blogimg/20210730.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.读入数据

```python
import pandas as pd

df = pd.read_csv("Region_Data.csv", encoding='gb2312', skiprows=3, skipfooter=2, engine="python")
```

`skiprows=3`表示读入文件时跳过前面三行；`skipfooter=2`表示读入文件时跳过后面两行。

数据见下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson24/24x1.png)

# 2.数据转换

将数据拆分为“地区”、“年份”和“生产总值”三个栏位：

```python
df = pd.melt(df, col_level=0, id_vars="地区")
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson24/24x2.png)

>`pandas.melt`的用法见本文2.1部分。

移除“年”：

```python
df['variable'] = df['variable'].map(lambda e: int(e.strip('年')))
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson24/24x3.png)

>`map`的用法见：[map](http://shichaoxin.com/2020/07/08/Python基础-第十四课-资料转换/#31map)。
>
>`strip`的用法见：[strip](http://shichaoxin.com/2019/05/14/Python基础-第五课-读写TXT文件/#241strip)。

修改字段名：

```python
df.columns = ['area', 'year', 'gross_product']
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson24/24x4.png)

## 2.1.`pandas.melt`

```python
def melt(
	frame, 
	id_vars=None, 
	value_vars=None, 
	var_name=None,
	value_name='value', 
	col_level=None
)
```

参数解释：

1. `frame`：要处理的数据集。
2. `id_vars`：不需要被转换的列名。
3. `value_vars`：需要转换的列名，如果剩下的列全部都要转换，就不用写了。
4. `var_name`和`value_name`：是自定义设置对应的列名。
5. `col_level`：如果列是MultiIndex，则使用此级别。

举个例子：

```python
import pandas as pd
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})
```

```
   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6
```

```python
pd.melt(df,id_vars=['A'])
#等同于：
pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
```

```
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
3  a        C      2
4  b        C      4
5  c        C      6
```

```python
pd.melt(df, id_vars=['A'], value_vars=['B'])
```

```
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
```

```python
pd.melt(df, id_vars=['A'], value_vars=['B'],
...         var_name='myVarname', value_name='myValname')
```

```
   A myVarname  myValname
0  a         B          1
1  b         B          3
2  c         B          5
```

multi-index的情况：

```python
df.columns = [list('ABC'), list('DEF')]
```

```
   A  B  C
   D  E  F
0  a  1  2
1  b  3  4
2  c  5  6
```

```python
pd.melt(df, col_level=0, id_vars=['A'], value_vars=['B'])
```

```
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
```

```python
pd.melt(df, id_vars=[('A', 'D')], value_vars=[('B', 'E')])
```

```
  (A, D) variable_0 variable_1  value
0      a          B          E      1
1      b          B          E      3
2      c          B          E      5
```

# 3.存储数据到数据库

```python
import pandas as pd
import sqlite3 as lite
with lite.connect('country_stat.sqlite') as db:
    df.to_sql('regional_gross_product', con=db, if_exists='replace', index=None)
```

# 4.筛选数据

读取表格（`*`表示选择所有字段）：

```sql
SELECT * FROM regional_gross_product
```

```python
with lite.connect('country_stat.sqlite') as db:
    df2 = pd.read_sql('SELECT * FROM regional_gross_product', con=db)
    print(df2.head())
```

```
     area  year  gross_product
0     北京市  2015       23014.59
1     天津市  2015       16538.19
2     河北省  2015       29806.11
3     山西省  2015       12766.49
4  内蒙古自治区  2015       17831.51
```

筛选行：

```sql
SELECT * FROM regional_gross_product WHERE year=2014
```

```python
with lite.connect('country_stat.sqlite') as db:
    df2 = pd.read_sql('SELECT * FROM regional_gross_product WHERE year=2014', con=db)
    print(df2.head())
```

```
     area  year  gross_product
0     北京市  2014       21330.83
1     天津市  2014       15726.93
2     河北省  2014       29421.15
3     山西省  2014       12761.49
4  内蒙古自治区  2014       17770.19
```

筛选字段：

```sql
SELECT area,gross_product FROM regional_gross_product
```

```python
with lite.connect('country_stat.sqlite') as db:
    df2 = pd.read_sql('SELECT area,gross_product FROM regional_gross_product', con=db)
    print(df2.head())
```

```
     area  gross_product
0     北京市       23014.59
1     天津市       16538.19
2     河北省       29806.11
3     山西省       12766.49
4  内蒙古自治区       17831.51
```

# 5.排序数据

排序数据（`DESC`为降序排列，默认为升序排列）：

```sql
SELECT * FROM regional_gross_product ORDER BY gross_product DESC
```

```python
with lite.connect('country_stat.sqlite') as db:
    df3 = pd.read_sql('SELECT * FROM regional_gross_product ORDER BY gross_product DESC', con=db)
    print(df3.head())
```

```
  area  year  gross_product
0  广东省  2015       72812.55
1  江苏省  2015       70116.38
2  广东省  2014       67809.85
3  江苏省  2014       65088.32
4  山东省  2015       63002.33
```

取得前三排行的数据：

```sql
SELECT * FROM regional_gross_product ORDER BY gross_product DESC LIMIT 3
```

```python
with lite.connect('country_stat.sqlite') as db:
    df3 = pd.read_sql('SELECT * FROM regional_gross_product ORDER BY gross_product DESC LIMIT 3', con=db)
    print(df3.head())
```

```
  area  year  gross_product
0  广东省  2015       72812.55
1  江苏省  2015       70116.38
2  广东省  2014       67809.85
```

# 6.聚合数据

使用`Group By`聚合数据：

```sql
SELECT area,AVG(gross_product) as avg_gross_product FROM regional_gross_product GROUP BY area
```

```python
#按area求分组平均
with lite.connect('country_stat.sqlite') as db:
    df4 = pd.read_sql('SELECT area,AVG(gross_product) as avg_gross_product FROM regional_gross_product GROUP BY area',
                      con=db)
    print(df4.head())
```

```
     area  avg_gross_product
0     上海市          17923.525
1     云南省           8531.537
2  内蒙古自治区          12403.454
3     北京市          15362.376
4     吉林省           9535.298
```

使用`Having`挑选出平均生产总值为10000以上的地区：

```sql
SELECT area,AVG(gross_product) as avg_gross_product FROM regional_gross_product GROUP BY area HAVING avg_gross_product >= 10000
```

```python
with lite.connect('country_stat.sqlite') as db:
    df4 = pd.read_sql(
        'SELECT area,AVG(gross_product) as avg_gross_product FROM regional_gross_product GROUP BY area HAVING avg_gross_product >= 10000',
        con=db)
    print(df4.head())
```

```
     area  avg_gross_product
0     上海市          17923.525
1  内蒙古自治区          12403.454
2     北京市          15362.376
3     四川省          19307.193
4     天津市          10408.911
```

# 7.代码地址

1. [SQL Query的使用](https://github.com/x-jeff/Python_Code_Demo/tree/master/Demo24)

# 8.参考资料

1. [Pandas 的melt的使用](https://blog.csdn.net/maymay_/article/details/80039677)
2. [pandas.melt](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.melt.html)