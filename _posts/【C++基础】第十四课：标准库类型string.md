---
layout:     post
title:      【C++基础】第十四课：标准库类型string
subtitle:   标准库string，定义和初始化string对象，string对象上的操作，处理string对象中的字符
date:       2020-02-10
author:     x-jeff
header-img: blogimg/20200210.jpg
catalog: true
tags:
    - C++ Series
---
>【C++基础】系列博客为参考[《C++ Primer中文版（第5版）》](https://www.phei.com.cn/module/goods/wssd_content.jsp?bookid=37655)（**C++11标准**）一书，自己所做的读书笔记。  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.标准库`<string>`

标准库类型`string`表示可变长的字符序列，使用`string`类型必须首先包含`string`头文件。

```c++
#include <string>
using std::string
```

# 2.定义和初始化`string`对象

初始化`string`对象最常用的一些方式：

```c++
string s1;//s1为空字符串
string s2=s1;//s2为空字符串
string s3(s1);//等价于string s3=s1
string s4="hello";
string s5("hello");//等价于string s5="hello"
string s6(10,'c');//s4的内容是cccccccccc
```

## 2.1.直接初始化和拷贝初始化

我们在[【C++基础】第八课：变量](http://shichaoxin.com/2019/05/20/C++基础-第八课-变量/)一文中的2.2部分介绍了C++中几种不同的初始化方式。

通过`string`我们可以清楚地看到在这些初始化方式之间到底有什么区别和联系。

❗️如果使用等号(=)初始化一个变量，实际上执行的是**拷贝初始化**，编译器把等号右侧的初始值拷贝到新创建的对象中去。

❗️如果不使用等号，则执行的是**直接初始化**。

当初始值只有一个时，使用直接初始化或拷贝初始化都行。如果像第2部分中的`s6`那样初始化要用到的值有多个，一般来说只能使用直接初始化的方式。

对于用多个值进行初始化的情况，非要用拷贝初始化的方式来处理也不是不可以，不过需要显示地创建一个（临时）对象用于拷贝：

```c++
string s7=string(10,'c');
```

# 3.`string`对象上的操作

![](https://github.com/x-jeff/BlogImage/raw/master/CPPSeries/Lesson14/14x1.png)

## 3.1.读写`string`对象

例如：

```c++
string s8;
cin>>s8;
cout<<s8<<endl;
```

⚠️在执行读取操作时，`string`对象会自动忽略开头的空白（即空格符、换行符、制表符等）并从第一个真正的字符开始读起，直到遇见下一处空白为止（即空格符、换行符、制表符等）。

例如输入的是`(空格)Hello3(空格).World!(空格)`，则输出将是`Hello3`，输出结果中没有任何空格。

再举一个例子：

```c++
string s1,s2;
cin>>s1>>s2;
cout<<s1<<s2<<endl;
```

例如输入的依旧是`(空格)Hello3(空格).World!(空格)`，则输出将是`Hello3.World!`。

## 3.2.读取未知数量的`string`对象

```c++
string word;
while(cin>>word)
	cout<<word<<endl;
```

例如可能得到结果（绿色字体为输入，白色字体为输出）：

![](https://github.com/x-jeff/BlogImage/raw/master/CPPSeries/Lesson14/14x2.png)

👉C++关于输入终止问题：

1. Windows系统：Enter➡️Ctrl+Z➡️Enter。
2. Linux系统：Ctrl+D。

## 3.3.使用getline读取一整行

👉有时我们希望能在最终得到的字符串中保留输入时的空白符，这时应该用`getline`函数代替原来的`>>`运算符。

❗️`getline`函数的参数是一个输入流和一个`string`对象，函数从给定的输入流中读入内容，直到遇到**换行符**为止（⚠️注意换行符也被读进来了），然后把所读的内容存入到那个`string`对象中去（⚠️注意不存换行符）。

❗️`getline`只要一遇到换行符就结束读取操作并返回结果，哪怕输入的一开始就是换行符也是如此。如果输入真的一开始就是换行符，那么所得的结果就是个空`string`。

```c++
string line;
while(getline(cin,line))
	cout<<line<<endl;
```

## 3.4.`string`的`empty`和`size`操作

👉`empty`函数根据`string`对象是否为空返回一个对应的布尔值。

```c++
//每次读入一整行，遇到空行直接跳过
while(getline(cin,line))
	if(!line.empty())
		cout<<line<<endl;
```

👉`size`函数返回`string`对象的长度（即`string`对象中字符的个数）。

```c++
string line;
//每次读入一整行，输出其中超过80个字符的行
while(getline(cin,line))
	if(line.size()>80)
		cout<<line<<endl;
```

## 3.5.`string::size_type`类型

⚠️对于1.2.4部分中的`size`函数来说，返回一个`int`或者`unsigned`似乎都是合情合理的。但其实`size`函数返回的是一个`string::size_type`类型的值。

`string`类及其他大多数标准库类型都定义了几种配套的类型。这些配套类型体现了标准库类型与机器无关的特性，类型`size_type`则是其中的一种。

⚠️尽管我们不太清楚`string::size_type`类型的细节，但有一点是肯定的：它是一个**无符号类型**的值而且能足够存放下任何`string`对象的大小。

❗️所有用于存放`string`类的`size`函数返回值的变量，都应该是`string::size_type`类型的。

```c++
string s9="abcd";
auto v1=s9.size();//v1=4
decltype(s9.size()) v2=-1;//v2=2**64-1=18446744073709551615
decltype(s9.size()) v3=1.3;//v3=1
```

⚠️假设n是一个具有负值的`int`，则表达式`s.size()<n`的判断结果几乎肯定是true。这是因为负值n会自动地转换成一个比较大的无符号值。

## 3.6.比较`string`对象

`string`类定义了几种用于比较字符串的运算符。这些运算符逐一比较`string`对象中的字符，并且对大小写敏感。

👉相等运算符(`==`和`!=`)分别检验两个`string`对象相等或不相等。`string`对象相等意味着它们的长度相同而且所包含的字符也全部相同。

👉关系运算符`<`、`<=`、`>`、`>=`分别检验一个`string`对象是否小于、小于等于、大于、大于等于另外一个`string`对象。

❗️比较规则：

1. 如果两个`string`对象的长度不同，而且较短`string`对象的每个字符都与较长`string`对象对应位置上的字符相同，就说较短`string`对象小于较长`string`对象。
2. 如果两个`string`对象在某些对应的位置上不一致，则`string`对象比较的结果其实是`string`对象中第一对相异字符比较的结果。

⚠️大写字母排在小写字母的前面，即"A"<"a"。其实单个字符的大小比较是根据[字符集](http://shichaoxin.com/2019/04/06/C++基础-第五课-基本内置类型/)中对应的数值，比如ACSII字符集。

## 3.7.两个`string`对象相加

```c++
string s1="hello,",s2="world\n";
string s3=s1+s2;//s3="hello,world\n"
s1+=s2;//s1=s1+s2="hello,world\n"
```

## 3.8.字面值和`string`对象相加

字面值可以和`string`对象相加，不过前提是该种类型可以自动转换成所需的类型。

标准库允许把**字符字面值**和**字符串字面值**转换成`string`对象。

```c++
string s1="hello",s2="world";
string s3=s1+","+s2+"\n";
```

⚠️当把`string`对象和字符字面值及字符串字面值混在一条语句中使用时，必须确保每个加法运算符（`+`）的两侧的运算对象至少有一个是`string`。

```c++
string s4=s1+",";//正确：把一个string对象和一个字面值相加
string s5="hello"+",";//错误：两个运算对象都不是string
string s6=s1+","+"world";//正确：每个加法运算符都有一个运算对象是string
string s7="hello"+","+s2;//⚠️错误：不能把字面值直接相加
```

‼️注意`s6`和`s7`是遵循从左向右累加的顺序。在`s6`中，`s1+","`得到一个`string`，然后再加上`"world"`，这是符合规则的。但是在`s7`中，`"hello"+","`加法运算符两侧均为字面值，不符合规则，所以错误。

‼️因为某些历史原因，也为了与C兼容，所以C++语言中的字符串字面值并不是标准库类型`string`的对象。切记，字符串字面值与`string`是不同的类型。

# 4.处理`string`对象中的字符

我们经常需要单独处理`string`对象中的字符，比如检查一个`string`对象是否包含空白，或者把`string`对象中的字母改成小写，再或者查看某个特定的字符是否出现等。

在`cctype`头文件中定义了一组标准库函数处理这部分工作：

![](https://github.com/x-jeff/BlogImage/raw/master/CPPSeries/Lesson14/14x3.png)

>⚠️**使用C++版本的C标准库头文件：**
>
>C++标准库中除了定义C++语言特有的功能外，也兼容了C语言的标准库。C语言的头文件形如`name.h`，C++则将这些文件命名为`cname`。也就是去掉了`.h`后缀，而在文件名`name`之前添加了字母c，这里的c表示这是一个属于C语言标准库的头文件。
>
>因此，`cctype`头文件和`ctype.h`头文件的内容是一样的，只不过从命名规范上来讲更符合C++语言的要求。❗️特别的，在名为`cname`的头文件中定义的名字从属于命名空间`std`，而定义在名为`.h`的头文件中的则不然。
>
>一般来说，C++程序应该使用名为`cname`的头文件而不使用`name.h`的形式，标准库中的名字总能在命名空间`std`中找到。如果使用`.h`形式的头文件，程序员就不得不时刻牢记哪些是从C语言那儿继承过来的，哪些又是C++语言所独有的。

## 4.1.处理每个字符

如果想对`string`对象中的每个字符做点儿什么操作，目前最好的办法是使用C++11新标准提供的一种语句：**范围for**语句。

```
for(declaration:expression)
	statement
```

其中，expression部分是一个对象，用于表示一个序列。declaration部分负责定义一个变量，该变量将被用于访问序列中的基础元素。每次迭代，declaration部分的变量会被初始化为expression部分的下一个元素值。

```c++
string str("some string");
for(auto c:str)
	cout<<c<<endl;
```

输出为：

![](https://github.com/x-jeff/BlogImage/raw/master/CPPSeries/Lesson14/14x4.png)

## 4.2.改变字符串中的字符

‼️如果想要改变`string`对象中字符的值，必须把循环变量定义成引用类型。

例如将字符串中的字符全部改为大写字母：

```c++
string s("Hello World!!!!");
for(auto &c:s)
	c=toupper(c);
cout<<s<<endl;//输出结果为HELLO WORLD!!!!
```

## 4.3.只处理一部分字符

如果要处理`string`对象中的每一个字符，使用范围for语句是个好主意。然而，有时我们需要访问的只是其中一个字符，或者访问多个字符但遇到某个条件就要停下来。

‼️要想访问`string`对象中的单个字符有两种方式：一种是**使用下标**，另外一种是**使用迭代器**。

‼️**下标运算符(`[]`)**接受的输入参数是`string::size_type`类型的值，这个参数表示要访问的字符的位置；返回值是该位置上字符的**引用**。

‼️`string`对象的下标从0计起。如果`string`对象s至少包含两个字符，则`s[0]`是第1个字符、`s[1]`是第2个字符、`s[s.size()-1]`是最后一个字符。

>`string`对象的下标必须大于等于0而小于`s.size()`。
>
>使用超出此范围的下标将引发不可预知的结果，以此推断，使用下标访问空`string`也会引起不可预知的结果。

下标的值称作**“下标”**或**“索引”**，任何表达式只要它的值是一个**整型值**就能作为索引。⚠️不过，如果某个索引是带符号类型的值将自动转换成由`string::size_type`表达的无符号类型。

```c++
if(!s.empty())
	cout<<s[0]<<endl;
```

在访问指定字符之前，首先检查s是否为空。其实不管什么时候只要对`string`对象使用了下标，都要确认在那个位置上确实有值。如果s为空，则`s[0]`的结果将是**未定义**的。

只处理第一个字符：

```c++
string s("some string");
if(!s.empty())
	s[0]=toupper(s[0]);
```

输出结果为：`Some string`。

## 4.4.使用下标执行迭代

```c++
string s("some string");
//依次处理s中的字符直至我们处理完全部字符或者遇到一个空白
for(decltype(s.size()) index=0;index!=s.size() && !isspace(s[index]);++index)
	s[index]=toupper(s[index]);
```

输出结果为：`SOME string`。

⚠️注意**逻辑与运算符(`&&`)**的使用。

## 4.5.使用下标执行随机访问

例如，想要编写一个程序把0到15之间的十进制数转换成对应的十六进制形式：

```c++
const string hexdigits="0123456789ABCDEF";
cout<<"Enter a series of numbers between 0 and 15"<<" separated by spaces. Hit ENTER when finished: "<<endl;
string result;
string::size_type n;
while(cin>>n)
	if(n<hexdigits.size())
		result+=hexdigits[n];
cout<<"Your hex number is: "<<result<<endl;
```

假设输入的内容为：`12 0 5 15 8 15`。程序的输出结果将是：`Your hex number is: C05F8F`。

# 5.代码地址

1. [标准库类型string](https://github.com/x-jeff/CPlusPlus_Code_Demo/tree/master/Demo14)