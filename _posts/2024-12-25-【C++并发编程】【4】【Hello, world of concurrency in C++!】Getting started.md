---
layout:     post
title:      【C++并发编程】【4】【Hello, world of concurrency in C++!】Getting started
subtitle:   thread，std::thread，join()
date:       2024-12-25
author:     x-jeff
header-img: blogimg/20191109.jpg
catalog: true
tags:
    - C++ Concurrency IN ACTION
---
>【C++并发编程】系列博客为参考《C++ Concurrency IN ACTION (SECOND EDITION)》一书，自己所做的读书笔记。  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Getting started

一个多线程C++程序和其他普通C++程序唯一的区别在于，有些函数可能会并发运行，因此我们需要确保共享数据在并发访问时是安全的。

# 2.Hello, Concurrent World

首先，下面是一个单线程的"Hello World"程序。

```c++
#include <iostream>
int main()
{
    std::cout<<"Hello World\n";
}
```

接下来我们启用一个独立的线程来打印信息。

```c++
#include <iostream>
#include <thread>
void hello()
{
    std::cout<<"Hello Concurrent World\n";
}
int main()
{
    std::thread t(hello);
    t.join();
}
```

用于管理线程的函数和类声明在`<thread>`中，而用于保护共享数据的功能则声明在其他头文件中。

每个线程都需要有一个初始函数（initial function），作为新线程执行的起点。对于一个程序的初始线程，这个初始函数是`main()`，而对于其他线程，则需要在`std::thread`对象的构造函数中指定初始函数。在上述代码示例中，名为`t`的`std::thread`对象将`hello()`函数作为它的初始函数。

因此，在上述代码示例中，一共有2个线程，一个是从`main()`开始的初始线程，另一个是从`hello()`开始的新线程。

新线程启动后，初始线程会继续执行。如果初始线程不等待新线程完成，它会直接继续执行到`main()`的末尾并结束程序，这可能会导致新线程还没有机会运行就被终止了。为了防止这种情况的发生，我们调用了`join()`。`join()`会使调用线程（即`main()`所在的初始线程）等待与`std::thread`对象（即`t`）关联的线程完成。