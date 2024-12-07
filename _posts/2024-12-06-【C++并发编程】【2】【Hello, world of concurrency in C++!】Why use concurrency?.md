---
layout:     post
title:      【C++并发编程】【2】【Hello, world of concurrency in C++!】Why use concurrency?
subtitle:   并发的优缺点
date:       2024-12-06
author:     x-jeff
header-img: blogimg/20221027.jpg
catalog: true
tags:
    - C++ Concurrency IN ACTION
---
>【C++并发编程】系列博客为参考《C++ Concurrency IN ACTION (SECOND EDITION)》一书，自己所做的读书笔记。  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Why use concurrency?

在应用程序中使用并发有两个主要原因：

* 关注点分离（separation of concerns）
* 性能（performance）

# 2.Using concurrency for separation of concerns

关注点分离指的是将相关的代码组织在一起，并将不相关的代码隔离开，这有助于提高代码的可读性和可维护性。而并发就允许将不同功能领域的操作彼此分离，即使这些操作需要同时执行。

比如电脑中的DVD播放器，这类应用程序有两大职责：

1. 从磁盘读取数据、解码图像和声音，并及时将这些数据发送到图形和音频硬件，以确保DVD播放不出现卡顿。
2. 处理用户输入，比如用户点击“暂停”、“返回菜单”或“退出”时的操作。

在单线程中，应用程序必须在播放期间定期检查用户输入，这将导致DVD播放代码与用户界面代码紧密交织在一起。而在多线程中，我们可以解耦DVD播放代码和用户界面代码，一个线程专注于处理用户界面，另一个线程负责DVD播放。此外，线程之间仍需一些交互，例如用户点击“暂停”时，但这些交互是与任务直接相关的。

# 3.Using concurrency for performance: task and data parallelism

任务并行性（task parallelism）指的是将一个复杂的任务拆分为多个子任务，每个子任务由一个线程独立完成。

数据并行性（data parallelism）指的是每个线程对不同的数据部分执行相同的操作。

# 4.When not to use concurrency

并发代码在很多情况下更难理解，编写和维护成本更高，如果并发带来的收益不够明显，则不建议使用并发。

此外，并发对性能的提升可能并没有预期的那么大。启动一个线程本身就存在固有的开销，因为操作系统需要分配相关的内核资源和栈空间，然后将新线程添加到调度器中，这一切都需要消耗时间。如果线程上运行的任务执行得很快，线程启动所需的时间可能远远大于任务的实际执行时间，使得整体性能反而更差。

此外，线程是一种有限的资源。如果一次运行太多线程，这会消耗操作系统的资源，并可能导致整个系统运行变慢。不仅如此，使用过多线程还可能耗尽进程的可用内存或地址空间，因为每个线程需要单独的栈空间。

最后，运行的线程越多（超过硬件支持能力时），操作系统需要进行的[上下文切换](http://shichaoxin.com/2024/09/30/C++并发编程-1-Hello,-world-of-concurrency-in-C++!-What-is-concurrency/#2concurrency-in-computer-systems)也就越多。每次上下文切换都需要消耗时间，而这些时间本可以用来执行有用的工作。因此，在某个点上，增加额外的线程会降低整体应用程序的性能，而不是提高性能。