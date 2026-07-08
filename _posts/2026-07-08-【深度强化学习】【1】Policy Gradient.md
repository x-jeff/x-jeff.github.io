---
layout:     post
title:      【深度强化学习】【1】Policy Gradient
subtitle:   Policy Gradient
date:       2026-07-08
author:     x-jeff
header-img: blogimg/20200220.jpg
catalog: true
tags:
    - Reinforcement Learning
---
>本文为参考李宏毅老师的"Deep Reinforcement Learning, 2018"课程所作的个人笔记。
>
>课程YouTube地址：[Deep Reinforcement Learning, 2018](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)。
>
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Policy Gradient

下图是RL的简要流程说明：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/1/1.png)

>关于RL的一些基础知识不在此详述，可回顾以下博客：
>
>* [【机器学习基础】第七十三课：[强化学习]任务与奖赏](https://shichaoxin.com/2026/05/09/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E4%B8%89%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E4%BB%BB%E5%8A%A1%E4%B8%8E%E5%A5%96%E8%B5%8F/)
>* [【机器学习基础】第七十四课：[强化学习]K-摇臂赌博机](https://shichaoxin.com/2026/05/12/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E5%9B%9B%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-K-%E6%91%87%E8%87%82%E8%B5%8C%E5%8D%9A%E6%9C%BA/)
>* [【机器学习基础】第七十五课：[强化学习]有模型学习](https://shichaoxin.com/2026/05/28/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E4%BA%94%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E6%9C%89%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0/)
>* [【机器学习基础】第七十六课：[强化学习]免模型学习](https://shichaoxin.com/2026/06/26/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E5%85%AD%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E5%85%8D%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0/)
>* [【机器学习基础】第七十七课：[强化学习]值函数近似](https://shichaoxin.com/2026/06/30/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E4%B8%83%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E5%80%BC%E5%87%BD%E6%95%B0%E8%BF%91%E4%BC%BC/)
>* [【机器学习基础】第七十八课：[强化学习]模仿学习](https://shichaoxin.com/2026/07/01/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E5%85%AB%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0/)

其中，Env、Actor、Reward都可以看作是一个function或者是一个模型，比如Env的输入是上一个状态$s_{n-1}$和选择的动作$a_{n-1}$，输出为下一个状态$s_n$；Actor的输入为当前环境状态$s_n$，输出为动作$a_n$；Reward的输入为状态$s_n$和动作$a_n$，输出得到的奖赏$r_n$。但需要注意的是，通常来说，我们无法控制Env和Reward的输出，只能优化Actor。Actor通过policy来接收输入并产生输出，通常用$\pi$来表示，比如policy可以是一个CNN，那policy的参数$\theta$就是CNN的网络权重。

>Env、Actor、Reward的输出都可能具有一定的随机性，即同样的输入，输出可能不同。

假如我们让RL去玩游戏，那么$s_n$就可看作是一帧游戏画面，$a_n$为模拟玩家基于当前画面做出的操作，Reward为玩家的得分。那么每玩一局游戏，就能产生一条轨迹：

$$\tau = \{ s_1,a_1,s_2,a_2,...,s_T,a_T \} \tag{1}$$

可以看到，这局游戏一共用了$T$步。那在当前Actor参数$\theta$的情况下，这条轨迹被采样得到的概率为：

$$\begin{align*} p_{\theta} (\tau) &= p(s_1)p_{\theta}(a_1 \mid s_1) p(s_2 \mid s_1,a_1) p_{\theta} (a_2 \mid s_2) p(s_3 \mid s_2, a_2) \cdots \\&= p(s_1) \prod_{t=1}^T p_{\theta} (a_t \mid s_t) p(s_{t+1} \mid s_t,a_t) \end{align*} \tag{2}$$

其中，$p(s_1)$和$p(s_{t+1} \mid s_t,a_t)$是Env产生的状态，其和Actor模型没有关系，所以没有下标$\theta$。

一条轨迹获得的总奖赏可表示为：

$$R(\tau) = \sum_{t=1}^T r_t \tag{3}$$

在一局游戏中，采样的轨迹并不是确定的，如果我们考虑所有可能的采样轨迹，则可得到这局游戏的期望奖赏为：

$$\bar{R}_{\theta} = \sum_{\tau} R(\tau) p_{\theta} (\tau) = E_{\tau \sim p_{\theta}(\tau)} [R(\tau)] \tag{4}$$

我们的优化目标就是通过更新$\theta$，使得期望奖赏$\bar{R}$的值越大越好，用到的优化方法就是policy gradient，其核心就是梯度上升法。因此我们首先需要计算$\bar{R}_{\theta}$的梯度：

$$\begin{align*} \nabla \bar{R}_{\theta} &= \sum_{\tau} R(\tau) \nabla p_{\theta} (\tau) \\&= \sum_{\tau} R(\tau) p_{\theta} (\tau) \frac{\nabla p_{\theta} (\tau)}{p_{\theta}(\tau)} \\&= \sum_{\tau} R(\tau) p_{\theta} (\tau) \nabla \log p_{\theta} (\tau) \\&= E_{\tau \sim p_{\theta}(\tau)} [ R(\tau) \nabla \log p_{\theta} (\tau) ] \\& \approx \frac{1}{N} \sum_{n=1}^N R(\tau ^n) \nabla \log p_{\theta} (\tau^n) \\&= \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} R (\tau^n) \nabla \log p_{\theta} (a_t^n \mid s_t^n)  \end{align*} \tag{5}$$

* 第2步：分子分母同时乘上$p_{\theta}(\tau)$。
* 第3步：根据公式$\nabla f(x) = f(x) \nabla \log f(x)$。
* 第5步：因为我们不太可能遍历所有可能的轨迹来求期望，所以采样有限个轨迹来近似。
* 第6步：对式(2)的两边同时取对数，基于公式$\log (ab) = \log a + \log b$可得，$\log p_{\theta} (\tau) = \log p(s_1) + \sum_{t=1}^T \log p_{\theta} (a_t \mid s_t) + \sum_{t=1}^T \log p(s_{t+1} \mid s_t, a_t)$，然后对参数$\theta$求梯度，因为第一项$\log p(s_1)$和第三项$\sum_{t=1}^T \log p(s_{t+1} \mid s_t, a_t)$与$\theta$无关，所以被消去，最终得到：$\nabla_{\theta} \log p_{\theta} (\tau) = \sum_{t=1}^T \nabla_{\theta} \log p_{\theta} (a_t \mid s_t)$。

使用梯度上升法更新参数$\theta$：

$$\theta \leftarrow \theta + \eta \nabla \bar{R}_{\theta} \tag{6}$$

每采集$N$条轨迹就更新一次$\theta$。

针对式(5)，如果选择一个动作，得到了正的奖赏，而选择另一个动作，却得到了负的奖赏，那么随着优化的进行，正奖赏动作的概率会增加，负奖赏动作的概率则会降低。但对于某些应用场景，奖赏始终是大于等于0的，比如在某个状态下，一共有3个可选择的动作$a,b,c$，其能得到的奖赏分别为$R_a=90,R_b=100,R_c=20$。对于优化算法来说，这3个动作都能使奖赏增加，都应该被鼓励。但因为3个动作被选择的概率之和为1，其中一个动作被选择的概率增加，就会导致另外两个动作被选择的概率降低。因为轨迹会被采样多次，假设这次采样选择了动作$c$，因为选择动作$c$也能使奖赏增加，因此在优化时，动作$c$的概率提升，相应的，真实的最优动作$b$的概率却被抑制下降了，这是我们不希望看到的。然后下一次可能采样到动作$b$，提升了动作$b$的概率，再下一次采样到了动作$a$，又提升了动作$a$的概率。这会导致更新方向随着随机采样来回波动，训练不稳定，难以收敛。因此我们对式(5)进一步修改，增加一个baseline项：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} (R (\tau^n)-b) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{7}$$

$b$可以是任意一个合理的值，比如$b \approx E[R(\tau)]$。接着上面的例子，假设我们把$b$设为$R_a,R_b,R_c$的平均值，那么动作$c$的奖赏$R_c-b=20-70=-50$就变成了负值，因此动作$c$不再会被鼓励。

在式(5)中，在一次轨迹采样中，对每个状态-动作对，其加权值都是$R (\tau^n)$，但这是不合理的，比如一条轨迹的最终奖赏并不高，但不意味着该条轨迹中的每个状态-动作对都不好；反之，对于一条最终奖赏很高的轨迹，其包含的状态-动作对也不都全是好的。因此，我们希望在一条轨迹中，每个状态-动作对的权重可以不一样，好的状态-动作对权重更高，差的状态-动作对权重更低。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/1/2.png)

我们通过上图的例子来说明，假设轨迹只包含3个状态-动作对。基于式(5)，对于第一条轨迹，其总奖赏为$5+0-2=3$，因此每个状态-动作对的权值就都是3；对于第二条轨迹，其总奖赏为$-5+0-2=-7$，因此每个状态-动作对的权值都是-7。但对于每个状态，在选择要执行的动作之后，其对前面已执行的状态-动作对没有任何影响，它只会影响后续状态-动作对的奖赏值，因于此理论，我们对每个状态-动作对的权重进行调整：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/1/3.png)

对于第一条轨迹，对于$(s_a,a_1)$，其权值为后续奖赏的总和，即$5+0-2=3$，同样的，对于$(s_b,a_2)$，其权值也为后续奖赏的总和，即$0-2=-2$，同理，$(s_c,a_3)$的权值为$-2$。第二条轨迹中，每个状态-动作对的权值计算方法也都是一样的。因此，我们可以将式(7)进一步优化为：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \left( \sum_{t'=t}^{T_n} r_{t'}^n -b \right) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{8}$$

此外，某个状态-动作对对后续的影响是衰减的，对离得近的状态影响更大，离得越远，影响越小，因此我们在式(8)中加上一个衰减系数：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} \left( \sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n -b \right) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{9}$$

其中，$\gamma < 1$。我们可以把$\left( \sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n -b \right)$简记为$A^{\theta}(s_t,a_t)$：

$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} A^{\theta}(s_t,a_t) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \tag{10}$$

$A^{\theta}(s_t,a_t)$表示在状态$s_t$下，相比选择其他动作，选择动作$a_t$，相对来说，好了多少。