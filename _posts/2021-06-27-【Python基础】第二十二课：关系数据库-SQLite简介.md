---
layout:     post
title:      【Python基础】第二十二课：关系数据库-SQLite简介
subtitle:   数据库，SQL，关系数据库，ACID原则，SQLite
date:       2021-06-27
author:     x-jeff
header-img: blogimg/20210627.jpg
catalog: true
tags:
    - Python Series
---
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.数据库

将数据以[结构化方式](http://shichaoxin.com/2019/03/26/Python基础-第四课-数据类型/#21结构化数据)做存储，让用户可以通过**结构化查询语言（Structured Query Language，简称SQL）**快速获取及维护数据。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson22/22x1.png)

👉**关系数据库（relational database）**：

1. 安全存储、管理数据。
	* 有效管理磁盘上的数据。
2. 保持数据的一致性。
	* ACID四原则。
		* 不可分割性（Atomicity）：交易必须全部完成或全部不完成。例如，转账。
		* 一致性（Consistency）：交易开始到结束，数据完整性都符合既设规则与限制。例如，账号。
		* 隔离性（Isolation）：并行的交易不会影响彼此。例如，余额查询。
		* 持久性（Durability）：进行完交易后，对数据库的变更会永久保留在数据库。例如，系统损毁。
3. 可以通过标准模型整合数据。
	* 使用SQL操作数据。

数据库中含多个数据表：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson22/22x2.png)

数据表包含的内容：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/PythonSeries/Lesson22/22x3.png)

# 2.SQLite

👉官网：[SQLite](https://www.sqlite.org/index.html)。

SQLite是一款轻型的数据库，是遵守ACID的关系型数据库管理系统。