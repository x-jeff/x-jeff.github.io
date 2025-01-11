---
layout:     post
title:      【C++并发编程】【5】【Managing threads】Basic thread management
subtitle:   join()，detach()，joinable()
date:       2025-01-01
author:     x-jeff
header-img: blogimg/20190806.jpg
catalog: true
tags:
    - C++ Concurrency IN ACTION
---
>【C++并发编程】系列博客为参考《C++ Concurrency IN ACTION (SECOND EDITION)》一书，自己所做的读书笔记。  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Basic thread management

每个C++程序至少有一个线程：即运行`main()`的线程。我们的程序随后可以启动其他线程，这些线程以另一个函数作为入口点。所有的这些线程会并发执行。就像程序在`main()`函数返回时退出一样，当指定的入口点函数返回时，线程也会退出。

# 2.Launching a thread

线程是通过构造一个`std::thread`对象来启动的，这个对象指定了在线程上运行的任务。

```c++
void do_some_work();
std::thread my_thread(do_some_work);
```

`std::thread`可以与任何可调用类型一起使用，因此我们也可以将一个具有[函数调用运算符](http://shichaoxin.com/2023/08/22/C++基础-第八十二课-重载运算与类型转换-函数调用运算符/#1函数调用运算符)的类的实例传递给`std::thread`构造函数。

```c++
class background_task
{
public:
    void operator() () const
    {
        do_something();
        do_something_else();
    }
};
background_task f;
std::thread my_thread(f);
```

在这种情况下，提供的函数对象会被复制到新创建的执行线程的存储空间中，并从那里调用。因此，必须确保复制的函数对象与原始对象的行为完全一致，否则结果可能会与预期不符。

但如果我们传递的是一个临时变量而不是命名变量，如下所示：

```c++
std::thread my_thread(background_task());
```

这相当于声明了一个名为`my_thread`的函数，这个函数有一个参数（即`background_task()`），该函数的返回类型为`std::thread`，所以上述代码并非是启动了一个线程。我们可以使用一对额外的括号或新统一的初始化语法来避免这个问题：

```c++
std::thread my_thread((background_task())); //使用额外的一对括号
std::thread my_thread{background_task()}; //使用新统一的初始化语法
```

使用[lambda表达式](http://shichaoxin.com/2022/12/13/C++基础-第五十八课-泛型算法-定制操作/#3lambda表达式)也能避免这个问题，如下所示：

```c++
std::thread my_thread([]{
    do_something();
    do_something_else();
});
```

启动线程后，我们必须决定是等待该线程完成还是让它独立运行。如果在`std::thread`对象被销毁之前都没有做出决定，`std::thread`对象在销毁时会调用`std::terminate()`，从而导致异常终止程序。需要注意的是，我们只需在`std::thread`对象被销毁前做出这个决定即可，即使此时线程可能已经完成了。此外，如果我们决定让线程独立运行，即使`std::thread`对象被销毁了，线程也将继续运行直至完成。

如果我们选择让线程独立运行，那么我们就需要确保线程所访问的数据在线程完成之前是有效的。

当线程函数持有指向局部变量的指针或引用时，如果线程在函数退出时尚未完成，就可能遇到这种问题，如下代码示例：

```c++
struct func
{
    int& i; //注意这里是引用
    func(int& i_) : i(i_) {} //注意这里是引用
    void operator() ()
    {
        for(unsigned j=0; j<1000000; ++j)
        {
            do_something(i);
        }
    }
};
void oops()
{
    int some_local_state = 0;
    func my_func(some_local_state); //对局部变量的引用
    std::thread my_thread(my_func);
    my_thread.detach();
}
```

因为我们调用了`detach()`让线程独立运行，所以函数`oops`结束后，线程`my_thread`可能仍在运行，但此时`some_local_state`这个局部变量已经被销毁了，因此此时调用`do_something(i)`将会访问一个已经被销毁的变量，从而导致未定义的行为。

# 3.Waiting for a thread to complete

我们可以通过调用`join()`来等待线程完成。

调用`join()`的同时也会清理与线程相关的任何存储空间，因此`std::thread`对象不再与已完成的线程关联，此时，它不再关联任何线程。这也意味着对一个线程只能调用一次`join()`，一旦调用过`join()`，该`std::thread`对象将不再是可连接的，此时调用`joinable()`会返回`false`。

# 4.Waiting in exceptional circumstances

如前所述，在`std::thread`对象销毁之前，我们必须确保调用了`join()`或`detach()`。如果我们打算让线程独立运行，那么在线程启动后立即调用`detach()`即可，这通常不会有什么问题。但我们如果打算等待线程完成，那我们就需要仔细选择调用`join()`的位置了，因为如果在线程启动后且在调用`join()`之前抛出了异常，那么`join()`的调用可能会被跳过。如下代码示例是一种解决办法：

```c++
struct func;
void f()
{
    int some_local_state = 0;
    func my_func(some_local_state);
    std::thread t(my_func);
    try 
    {
        do_something_in_current_thread();
    }
    catch(...)
    {
        t.join();
        throw; //函数f退出，并将该异常抛给上一层，即函数f的调用者
    }
    t.join();
}
```

但是使用[`try/catch`](http://shichaoxin.com/2021/11/19/C++基础-第三十三课-try语句块和异常处理/)显得比较繁琐，且容易出错，比如可能会遗漏某些退出路径，导致线程未被正确管理。

因此另一种简化的方法是RAII（Resource Acquisition Is Initialization），比如提供一个在其析构函数中调用`join()`的类，如下代码所示：

```c++
class thread_guard
{
    std::thread& t;
public:
    explicit thread_guard(std::thread& t_) : t(t_) {}
    ~thread_guard()
    {
        if(t.joinable())
        {
            t.join();
        }
    }
    thread_guard(thread_guard const&) = delete;
    thread_guard& operator=(thread_guard const&) = delete;
};
struct func;
void f()
{
    int some_local_state = 0;
    func my_func(some_local_state);
    std::thread t(my_func);
    thread_guard g(t);
    do_something_in_current_thread();
}
```

当当前线程执行到函数`f`的末尾时，局部对象会按照与其构造顺序相反的顺序销毁。因此，`thread_guard`对象`g`会最先被销毁，并且在析构函数中调用`join()`。即使函数`f`因为`do_something_in_current_thread`抛出了异常而退出，这一过程也会发生。

`thread_guard`的析构函数在调用`join()`之前，首先测试了`std::thread`对象是否是`joinable()`的。这是非常重要的，因为对于某个线程的执行，`join()`只能被调用一次。如果线程已经被`join()`过，再次调用`join()`将会是一个错误。

[拷贝构造函数](http://shichaoxin.com/2023/04/24/C++基础-第六十九课-拷贝控制-拷贝-赋值与销毁/#2拷贝构造函数)和[拷贝赋值运算符](http://shichaoxin.com/2023/04/24/C++基础-第六十九课-拷贝控制-拷贝-赋值与销毁/#3拷贝赋值运算符)被标记为`=delete`，以确保编译器不会自动生成它们。拷贝或赋值这样的对象是危险的，因为它可能会在所管理的线程超出作用域后仍然存在。通过将这些函数声明为`delete`，任何试图拷贝`thread_guard`对象的操作都会导致编译错误。

# 5.Running threads in the background

在`std::thread`对象调用`detach()`后，线程将在后台运行（run in the background），与主线程之间不再有直接的通信手段。此时已经无法等待该线程完成；一旦线程被`detach()`，就无法再获得与该线程关联的`std::thread`对象，因此也无法再对其进行`join()`操作。`detach()`的线程在后台运行后，其所有权和控制权被交给了C++ Runtime Library，C++ Runtime Library会确保线程退出时正确回收其相关资源。

`detach()`的线程通常被称为守护线程（daemon threads）。

```c++
std::thread t(do_background_work);
t.detach();
assert(!t.joinable());
```

我们不能对没有关联执行线程的`std::thread`对象调用`detach()`。这与`join()`的要求完全相同，只有当`t.joinable()`返回`true`时，才能对`std::thread`对象`t`调用`t.detach()`。