---
layout:     post
title:      【深度强化学习】【2】Proximal Policy Optimization
subtitle:   off-policy，TRPO，PPO，PPO2
date:       2026-07-28
author:     x-jeff
header-img: blogimg/20220816.jpg
catalog: true
tags:
    - Reinforcement Learning
---
>本文为参考李宏毅老师的"Deep Reinforcement Learning, 2018"课程所作的个人笔记。
>
>课程YouTube地址：[Deep Reinforcement Learning, 2018](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)。
>
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.on-policy vs off-policy

>关于on-policy和off-policy的另一篇博文：[【机器学习基础】第七十六课：[强化学习]免模型学习](https://shichaoxin.com/2026/06/26/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E4%B8%83%E5%8D%81%E5%85%AD%E8%AF%BE-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E5%85%8D%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0/)。

简单来说，如果和环境互动的agent和我们要训练的agent是同一个，那就是on-policy。如果和环境互动的agent和我们要训练的agent不是同一个，就是off-policy。

我们在[【深度强化学习】【1】Policy Gradient](https://shichaoxin.com/2026/07/08/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-1-Policy-Gradient/)中介绍的属于on-policy。根据[这里的式(5)](https://shichaoxin.com/2026/07/08/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-1-Policy-Gradient/)，可以知道，在on-policy中，基于参数$\theta$，我们需要采样多个轨迹$\tau$，然后才能更新一次$\theta$，并且更新后，我们要基于新的$\theta$，重新采样多个轨迹，之后才能继续更新参数$\theta$。因此，我们可以用off-policy，即使用固定参数的agent与环境互动并采集数据，以此来更新目标agent的参数。这样就不用每次更新参数时都重新采集数据了。

# 2.Importance Sampling

期望可做如下近似：

$$E_{x \sim p} [f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x^i) \tag{1}$$

其中，数据$x^i$采样自分布$p$。假设我们无法从分布$p$中采样$x^i$，却只能从另一个分布$q$中采样数据$x^i$，此时我们该如何计算式(1)呢？

$$\begin{align*} E_{x\sim p} [f(x) ] &= \int f(x) p(x) dx \\&= \int f(x) \frac{p(x)}{q(x)} q(x) dx \\&= E_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right] \end{align*} \tag{2}$$

>个人注解：“采样”指的是按照某种分布产生样本，无法从分布$p$中采样或不容易从分布$p$中采样不等同于无法计算$p(x)$。比如存在以下可能性：能采样但不知道$p(x)$、能计算$p(x)$但很难采样。

接下来，我们根据公式$VAR[X]=E[X^2]-(E[X])^2$来分别计算$[f(x) ]$和$\left[ f(x) \frac{p(x)}{q(x)} \right]$的方差。

$$Var _{x \sim p} [f(x) ] = E_{x \sim p} [f(x)^2] - (E_{x \sim p} [f(x)]) ^2 \tag{3}$$

$$\begin{align*} Var _{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right] &= E_{x \sim q} \left[ \left( f(x) \frac{p(x)}{q(x)} \right)^2 \right] - \left( E_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right] \right)^2 \\&= \int \left( f(x) \frac{p(x)}{q(x)} \right)^2 q(x) dx - \left( \int f(x) \frac{p(x)}{q(x)} q(x) dx \right)^2 \\&= \int f(x)^2 \frac{p(x)}{q(x)} p(x) dx - \left( \int f(x) p(x) dx \right)^2 \\&= E_{x \sim p} \left[ f(x)^2 \frac{p(x)}{q(x)} \right] - (E_{x \sim p} [f(x)])^2 \end{align*} \tag{4}$$

从式(3)和式(4)可以看出，虽然$[f(x) ]$和$\left[ f(x) \frac{p(x)}{q(x)} \right]$的期望一样，但如果分布$p$和分布$q$的差异过大，会使得$\left[ f(x) \frac{p(x)}{q(x)} \right]$的方差变大，这就导致每次用有限样本算出来的值可能波动很大，如果要让平均结果接近真实期望，就需要更大的样本量，采样成本会变高。方差过大也会导致梯度更新不稳定。因此，通常情况下，分布$p$和分布$q$的差异不能太大。

# 3.off-policy

根据[这里的式(5)](https://shichaoxin.com/2026/07/08/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-1-Policy-Gradient/)，结合第2部分，可以推导出off-policy的梯度计算：

$$\nabla \bar{R}_{\theta} = E_{\tau \sim p_{\theta '}(\tau)} \left[ \frac{p_{\theta} (\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_{\theta} (\tau) \right] \tag{5}$$

其中，$\theta$是我们要训练的目标agent，$\theta'$是与环境互动、用来采样轨迹的agent。

根据[这里的式(10)](https://shichaoxin.com/2026/07/08/%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-1-Policy-Gradient/)，off-policy的梯度更新策略可表示为：

$$\begin{align*} &\quad E_{(s_t,a_t) \sim \pi_{\theta}} [A^{\theta}(s_t,a_t) \nabla \log p_{\theta} (a_t^n \mid s_t^n) ] \\&=  E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{P_{\theta} (s_t,a_t)}{P_{\theta '} (s_t,a_t)} A^{\theta}(s_t,a_t) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \right] \\& \approx E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{P_{\theta} (s_t,a_t)}{P_{\theta '} (s_t,a_t)} A^{\theta'}(s_t,a_t) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \right] \\&= E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta} (a_t \mid s_t) p_{\theta} (s_t)}{p_{\theta '} (a_t \mid s_t) p_{\theta'} (s_t)} A^{\theta'}(s_t,a_t) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \right] \\&= E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta} (a_t \mid s_t) }{p_{\theta '} (a_t \mid s_t) } A^{\theta'}(s_t,a_t) \nabla \log p_{\theta} (a_t^n \mid s_t^n) \right] \end{align*} \tag{6}$$

* 第2步：因为在off-policy中，是$\theta'$在采样，并与环境互动，所以此处将$A^{\theta}$改为$A^{\theta'}$。
* 第3步：全概率展开。
* 第4步：因为无论是$\theta$还是$\theta'$，对状态$s_t$的出现都没什么影响，所以我们可以认为$p_{\theta} (s_t)$和$p_{\theta'} (s_t)$是差不多的，因此把这一项消去。

式(6)是梯度的计算，可以根据式(6)反推出目标函数的形式为：

$$J^{\theta'} (\theta) = E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta} (a_t \mid s_t)}{p_{\theta'} (a_t \mid s_t)} A^{\theta'} (s_t,a_t) \right] \tag{7}$$

因为我们要优化的是$\theta$，但与环境互动的是$\theta'$，所以我们这里写为$J^{\theta'} (\theta)$。我们可以验证下，对式(7)求导，可以得到式(6)：

$$\begin{align*} \nabla_{\theta} J^{\theta'} (\theta) &= \nabla _{\theta} E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta} (a_t \mid s_t)}{p_{\theta'} (a_t \mid s_t)} A^{\theta'} (s_t,a_t) \right] \\&= E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{1}{p_{\theta'}(a_t \mid s_t)} A^{\theta'} (s_t,a_t) \nabla_{\theta} p_{\theta} (a_t \mid s_t) \right] \\&= E_{(s_t,a_t) \sim \pi_{\theta'}} \left[ \frac{p_{\theta} (a_t \mid s_t) }{p_{\theta '} (a_t \mid s_t) } A^{\theta'}(s_t,a_t) \nabla \log p_{\theta} (a_t \mid s_t) \right] \end{align*} \tag{8}$$

* 第2步：因为只有$p_{\theta} (a_t \mid s_t)$是和$\theta$相关的，所以对$\theta$求导，只针对这一项即可。
* 第3步：根据公式$\nabla f(x) = f(x) \nabla \log f(x)$，有$\nabla_{\theta} p_{\theta} (a_t \mid s_t) = p_{\theta} (a_t \mid s_t) \nabla \log p_{\theta} (a_t \mid s_t)$。

# 4.PPO

基于第2部分，我们知道$p_{\theta}$和$p_{\theta'}$不能差太多，因此我们对式(7)加一个约束项：

$$J_{PPO}^{\theta'} (\theta) = J^{\theta'}(\theta) - \beta \text{KL}(\theta,\theta') \tag{9}$$

其中，$\text{KL}$是[KL散度](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)，用于衡量分布$\theta$和分布$\theta'$之间的相似程度。式(9)就是**Proximal Policy Optimization（PPO）**的目标函数。而PPO的前身**Trust Region Policy Optimization（TRPO）**则是把[KL散度](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)单独提取出来作为限制条件：

$$J_{TRPO}^{\theta'}(\theta) = J^{\theta'} (\theta), \quad KL(\theta, \theta') < \delta \tag{10}$$

虽然PPO和TRPO的性能差不多，但PPO在实现上更为容易。PPO算法的整体步骤见下：

* 初始化目标参数$\theta$。
* 对于每次迭代：
    * 使用$\theta^k$与环境交互并采集数据$\\{ s_t,a_t \\}$，同时计算$A^{\theta^k} (s_t,a_t)$。
    * 使用式(9)对目标参数$\theta$进行优化：$J_{PPO}^{\theta^k} (\theta) = J^{\theta^k} (\theta) - \beta \text{KL}(\theta, \theta^k)$。其中，在实际计算时，$J^{\theta^k}(\theta)$可近似为：$J^{\theta^k} (\theta) \approx \sum_{(s_t,a_t)} \frac{p_{\theta} (a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)} A^{\theta^k} (s_t, a_t)$。注意：使用上一步采集到的数据，这一步可以做多次优化。

额外的，还可以对[KL散度](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)项设置一个最大值$\text{KL}_{max}$和最小值$\text{KL}_{min}$，当$\text{KL}(\theta, \theta^k) > \text{KL}_{max}$时，增加$\beta$（即约束力度太小，需要加强）；当$\text{KL}(\theta, \theta^k) < \text{KL}_{min}$时，减小$\beta$（即约束力度太大，需要削弱）。

为了简化掉[KL散度](https://shichaoxin.com/2021/10/30/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Generative-Adversarial-Nets/#9kl%E6%95%A3%E5%BA%A6)，又提出了PPO2算法：

$$J_{PPO2}^{\theta^k} (\theta) \approx \sum_{(s_t,a_t)} \min \left( \frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)} A^{\theta^k}(s_t,a_t), \text{clip} \left( \frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}, 1-\epsilon,1+\epsilon \right) A^{\theta^k}(s_t,a_t) \right) \tag{11}$$

用下图解释下式(11)，两个子图的横轴都是$\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}$，蓝色线表示函数$f(\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}) = \text{clip} \left( \frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}, 1-\epsilon,1+\epsilon \right)$，绿色线表示函数$f(\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}) = \frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}$。当$A^{\theta^k}(s_t,a_t) > 0$时，表示动作可以带来正收益，我们希望该动作出现的概率$p_{\theta}(a_t \mid s_t)$越大越好，但我们并不能一味的增加$p_{\theta}(a_t \mid s_t)$，因为这会导致分布$\theta$和分布$\theta^k$的差异越来越大，即$\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}$越来越大，因此见下图左中的红线（取min的效果），当$\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}$大于$1+\epsilon$时，再增加$p_{\theta}(a_t \mid s_t)$就不能带来额外的收益了。反之，如果$A^{\theta^k}(s_t,a_t) < 0$，则表示该动作会带来负收益，我们就希望该动作被选择的概率越低越好，即$p_{\theta}(a_t \mid s_t)$越小越好，类似的，见下图右中的红线（取min的效果），我们也不能一味的减小$p_{\theta}(a_t \mid s_t)$，当$\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta^k}(a_t \mid s_t)}$小于$1-\epsilon$时，再减小$p_{\theta}(a_t \mid s_t)$就没有意义了，并不能带来更多的收益。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/ReinforcementLearning/DRL/2/1.png)