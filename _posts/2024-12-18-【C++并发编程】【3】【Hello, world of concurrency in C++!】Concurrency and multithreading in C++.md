---
layout:     post
title:      【C++并发编程】【3】【Hello, world of concurrency in C++!】Concurrency and multithreading in C++
subtitle:   多线程技术在C++中的发展
date:       2024-12-18
author:     x-jeff
header-img: blogimg/20220731.jpg
catalog: true
tags:
    - C++ Concurrency IN ACTION
---
>【C++并发编程】系列博客为参考《C++ Concurrency IN ACTION (SECOND EDITION)》一书，自己所做的读书笔记。  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Concurrency and multithreading in C++

从C++11标准开始，才正式支持开发者编写不依赖平台（比如Windows，Linux等）的多线程代码。

# 2.History of multithreading in C++

1998年的C++标准还未承认线程的存在。但许多C++编译器供应商通过各种平台特定的扩展来支持多线程。C++程序员们也通过类库（比如MFC、Boost、ACE）来提供更高级的多线程支持，这些类库封装了平台特定的API，简化了开发任务。

# 3.Concurrency support in the C++11 standard

这一切随着C++11标准的发布而改变。C++11不仅引入了一个线程感知的内存模型，而且C++标准库也得到了扩展，包含了以下类：

* 用于管理线程的类。
* 用于保护共享数据的类。
* 用于在线程之间同步操作的类。
* 用于低级别原子操作的类。

# 4.More support for concurrency and parallelism in C++14 and C++17

C++14新增了一个用于保护共享数据的新型互斥锁。C++17则引入了一整套并行算法，为高性能并行计算提供了极大的便利。

# 5.Efficiency in the C++ Thread Library

抽象开销（abstraction penalty）是高性能计算中开发者需要考虑的重要因素，特别是在使用C++标准库的高级功能时。C++标准委员会在设计标准线程库时，也充分考虑了抽象开销的问题，使得开发者在追求性能的同时，还能享受到标准库带来的易用性和跨平台支持。C++标准库提供了功能丰富的高级工具，但在极端性能场景下，这些工具的额外功能可能带来一定的性能成本。尽管手动实现底层功能可能减少一些开销，但其复杂性和错误风险通常大于性能收益。在大多数情况下，性能瓶颈更多来自于程序设计（如锁争用问题），而不是工具本身的性能。在极少数情况下，当C++标准库无法提供所需的性能或行为时，可能需要使用平台特定的工具或功能。

# 6.Platform-specific facilities

尽管C++线程库为多线程和并发提供了相当全面的工具，但在某些平台上，可能会有超出其范围的特定功能。为了在不放弃使用C++标准线程库所带来的好处的情况下方便地访问这些平台特定功能，C++线程库中的一些类型可能提供一个`native_handle()`成员函数，允许直接使用底层实现并通过平台特定的API进行操作。