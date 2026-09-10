---
layout:     post
title:      【深度强化学习】【4】Actor-Critic
subtitle:   A2C，A3C，Pathwise Derivative Policy Gradient
date:       2026-09-10
author:     x-jeff
header-img: blogimg/20210717.jpg
catalog: true
tags:
    - Reinforcement Learning
---
>本文为参考李宏毅老师的"Deep Reinforcement Learning, 2018"课程所作的个人笔记。
>
>课程YouTube地址：[Deep Reinforcement Learning, 2018](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)。
>
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Asynchronous Advantage Actor-Critic

在介绍[Policy Gradient](https://shichaoxin.com/2026/07/08/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-1-Policy-Gradient/)时，我们使用下式来计算梯度：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \left( \sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n -b \right) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{1}$$

我们将其简记为：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \left( G_t^n -b \right) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{2}$$

因为随机性的存在，给定同样的状态$s$和动作$a$，得到的累积奖赏值可能是不同的，所以说$G_t^n$是不稳定的，如下图所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/4/1.png)

因为我们是通过采样来近似期望，所以采样数量越多，近似越准确。但在实际实现时，由于成本等原因，采样通常是有限的，所以$G_t^n$在有限次采样中得到的值可能很不稳定，差异很大，这会影响训练的稳定性。那有没有一种稳定的方法来直接预测$G_t^n$的期望值呢？其实，$G_t^n$的期望是和[Q函数](https://shichaoxin.com/2026/05/28/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E4%BA%94%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E6%9C%89%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0/#2%E7%AD%96%E7%95%A5%E8%AF%84%E4%BC%B0)的定义是一致的：

$$E[G_t^n] = Q^{\pi_{\theta}} (s_t^n, a_t^n) \tag{3}$$

此外，我们可以把$b$设置为[V函数](https://shichaoxin.com/2026/05/28/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E4%BA%94%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E6%9C%89%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0/#2%E7%AD%96%E7%95%A5%E8%AF%84%E4%BC%B0)：

$$b = V^{\pi_{\theta}} (s_t^n) \tag{4}$$

此时，式(2)就变为：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \left( Q^{\pi_{\theta}} (s_t^n, a_t^n) -V^{\pi_{\theta}} (s_t^n) \right) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{5}$$

式(5)就是Actor-Critic。但是式(5)存在一个风险就是我们需要训练两个critic网络模型，一个Q网络模型，一个V网络模型，由网络预测带来的误差可能会加倍。那能不能只训练一个critic网络模型呢？答案是可以的，Q函数可以通过下式转换为V函数：

$$Q^{\pi} (s_t^n,a_t^n) = E[ r_t^n + V^{\pi}(s_{t+1}^n) ] \tag{6}$$

式(6)右边之所以要求期望，是因为$r_t^n$是一个随机变量，只有求期望，等式才能成立。为了简化，作者也通过实验进行了评估，我们可以拿掉式(6)中的期望：

$$Q^{\pi} (s_t^n,a_t^n) =  r_t^n + V^{\pi}(s_{t+1}^n) \tag{7}$$

将式(7)代入式(5)，得到：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \left( r_t^n + V^{\pi}(s_{t+1}^n) -V^{\pi} (s_t^n) \right) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{8}$$

依据式(8)，我们只需训练一个V网络模型即可。其实式(8)中的$r_t^n$是一个随机变量，但相比式(2)中的$G_t^n$，其随机性要小很多，影响也比较小。式(8)就是Advantage Actor-Critic，即A2C。整个A2C的训练流程见下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/4/2.png)

整个流程中会涉及两个网络模型，一个是actor网络模型，其输入为状态$s$，输出为动作的概率分布，另一个是critic网络模型，即我们之前提到的式(8)中的V网络模型，其输入也是状态$s$，输出为$V^{\pi}(s)$。既然actor网络模型和critic网络模型的输入都是一样的，我们就可以把两个模型捏在一起，如下图所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/4/3.png)

因为A2C训练起来比较慢，所以我们可以多个A2C一起训练，这就是所谓的Asynchronous Advantage Actor-Critic，即A3C，如下图所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/4/4.png)

每个worker network和global netowork都是一样的A2C结构。每个worker都从global network拷贝参数，并独自进行训练，之后将计算好的梯度传回给global network。对于global network，不论是哪个worker传回来的梯度，它都基于现在的参数进行梯度更新。

# 2.Pathwise Derivative Policy Gradient

我们在介绍[Q-learning](https://shichaoxin.com/2026/09/03/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-3-Q-learning/#3continuous-action)时提到过，因为其核心计算为$a = \underset{a}{\text{arg max}} Q(s,a)$，所以它不擅长解决continuous action。而pathwise derivative policy gradient的解决思路是，构建一个actor网络模型用于预测最优动作，并把这个最优动作喂给Q网络模型：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/4/5.png)

在绿色部分训练Q网络模型时，actor模型固定；在蓝色部分训练actor时，Q网络模型固定。我们可以基于[原始的Q-learning算法](https://shichaoxin.com/2026/09/03/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-3-Q-learning/#1basic-idea)加以修改，得到pathwise derivative policy gradient算法的流程：

* 初始化两个critic用于近似Q函数：$Q$和$\hat{Q}$，并且让$\hat{Q} = Q$。同时初始化两个actor，让$\hat{\pi} = \pi$。
* 对于每次迭代：
    * 对于每个时间步$t$：
        * 基于actor模型，给定状态$s_t$，得到动作$a_t$。
        * 得到奖赏$r_t$，到达新状态$s_{t+1}$。
        * 将$(s_t,a_t,r_t,s_{t+1})$存入buffer。
        * 从buffer中取出数据$(s_i,a_i,r_i,s_{i+1})$（通常是取一个batch的数据）。
        * 求$y = r_i + \hat{Q} (s_{i+1}, \hat{\pi} (s_{i+1}))$。
        * 更新第一个模型$Q$的参数，使$Q(s_i,a_i)$接近$y$。
        * 更新模型$\pi$的参数，最大化$Q(s_i, \pi (s_i))$。
        * 每经过$C$步，就将第二个模型$\hat{Q}$更新为第一个模型$Q$，即执行$\hat{Q} = Q$。
        * 每经过$C$步，执行$\hat{\pi} = \pi$。

pathwise derivative policy gradient也可看作是一种actor-critic方法。