---
layout:     post
title:      【C++基础】第四十九课：[IO库]string流
subtitle:   string流，sstream，istringstream，ostringstream，stringstream
date:       2022-08-16
author:     x-jeff
header-img: blogimg/20220816.jpg
catalog: true
tags:
    - C++ Series
---
>【C++基础】系列博客为参考[《C++ Primer中文版（第5版）》](https://www.phei.com.cn/module/goods/wssd_content.jsp?bookid=37655)（**C++11标准**）一书，自己所做的读书笔记。  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.string流

sstream头文件定义了三个类型来支持内存IO，这些类型可以向string写入数据，从string读取数据，就像string是一个IO流一样。

**istringstream**从string读取数据，**ostringstream**向string写入数据，而头文件**stringstream**既可从string读数据也可向string写数据。与[fstream类型](http://shichaoxin.com/2022/08/09/C++基础-第四十八课-IO库-文件输入输出/)类似，头文件sstream中定义的类型都继承自我们已经使用过的[iostream](http://shichaoxin.com/2022/07/31/C++基础-第四十七课-IO库-IO类/)头文件中定义的类型。除了继承得来的操作，sstream中定义的类型还增加了一些成员来管理与流相关联的string。下表列出了这些操作，可以对stringstream对象调用这些操作，但不能对其他IO类型调用这些操作。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/CPPSeries/Lesson49/49x1.png)

>[explicit的用法](http://shichaoxin.com/2022/07/13/C++基础-第四十五课-类-构造函数再探/#43抑制构造函数定义的隐式转换)。

# 2.使用istringstream

考虑这样一个例子，假定有一个文件，列出了一些人和他们的电话号码。某些人只有一个号码，而另一些人则有多个——家庭电话、工作电话、移动电话等。我们的输入文件看起来可能是这样的：

```
morgan 2015552368 8625550123
drew 9735550130
lee 6095550132 2015550175 8005550000
```

文件中每条记录都以一个人名开始，后面跟随一个或多个电话号码。我们首先定义一个简单的类来描述输入数据：

```c++
//成员默认为公有
struct PersonInfo {
	string name;
	vector<string> phones;
};
```

读取数据文件的程序见下：

```c++
string line, word; //分别保存来自输入的一行和单词
vector<PersonInfo> people; //保存来自输入的所有记录
//逐行从输入读取数据，直至cin遇到文件尾（或其他错误）
while (getline(cin, line)) {
	PersonInfo info; //创建一个保存此记录数据的对象
	istringstream record(line); //将记录绑定到刚读入的行
	record >> info.name; //读取名字
	while (record >> word) //读取电话号码
		info.phones.push_back(word);
	people.push_back(info);
}
```

# 3.使用ostringstream

当我们逐步构造输出，希望最后一起打印时，ostringstream是很有用的。例如，在第2部分的例子中，我们可能想逐个验证电话号码并改变其格式。如果所有号码都是有效的，我们希望输出一个新的文件，包含改变格式后的号码。对于那些无效的号码，我们不会将它们输出到新文件中，而是打印一条包含人名和无效号码的错误信息。

由于我们不希望输出有无效电话号码的人，因此对每个人，直到验证完所有电话号码后才可以进行输出操作。但是，我们可以先将输出内容“写入”到一个内存ostringstream中：

```c++
for (const auto &entry : people) {
	ostringstream formatted, badNums;
	for (const auto &nums : entry.phones) {
		if (!valid(nums)) {
			badNums << " " << nums;
		} else
			formatted << " " << format(nums);
	}
	if (badNums.str().empty())
		os << entry.name << " " << formatted.str() << endl;
	else
		cerr << "input error: " << entry.name << " invalid number(s) " << badNums.str() << endl;
}
```

在此程序中，我们假定已有两个函数，valid和format，分别完成电话号码验证和改变格式的功能。