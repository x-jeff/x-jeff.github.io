---
layout:     post
title:      【论文阅读】Attention Is All You Need
subtitle:   Transformer，Multi-Head Attention
date:       2022-03-26
author:     x-jeff
header-img: blogimg/20220326.jpg
catalog: true
tags:
    - Natural Language Processing
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)，尤其是[LSTM](http://shichaoxin.com/2020/12/09/深度学习基础-第四十二课-GRU和LSTM/#3lstm)，已经成为了[序列模型](http://shichaoxin.com/2020/11/08/深度学习基础-第三十九课-序列模型/)和转化问题（如机器翻译）的最优方法。此后的很多研究也都致力于推动循环语言模型和编码-解码框架的发展。

[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)固有的序列属性阻碍了训练的并行化。虽然后续有些研究通过一些方法提升了模型计算效率，但是序列属性固有的限制依然存在。

此外，[注意力机制](http://shichaoxin.com/2021/03/09/深度学习基础-第四十八课-注意力模型/)也逐渐成为很多序列模型和转化模型中不可或缺的一部分，其通常搭配[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)一起使用。

因此本文提出一种新的框架：Transformer。完全抛弃[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)那一套理论，只依靠注意力机制来构建输入和输出之间的全局依赖关系。并且，Transformer可以很好的实现并行化且在翻译质量上达到了SOTA的水平。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/3.png)

# 2.Background

据我们所知，Transformer是第一个完全依靠self-attention机制的转化模型（没有使用[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)和[CNN](http://shichaoxin.com/2020/07/04/深度学习基础-第二十八课-卷积神经网络基础/)）。

# 3.Model Architecture

大多数序列转化模型都使用了编码-解码结构。首先，编码器将输入序列$(x_1,...,x_n)$转化为$\mathbf{z}=(z_1,...,z_n)$。然后，解码器再将$\mathbf{z}$转化为输出序列$(y_1,...,y_m)$。

Transformer的总体框架见下（左半部分为编码器，右半部分为解码器）：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/1.png)

## 3.1.Encoder and Decoder Stacks

👉**Encoder:**

编码器由6个相同的层构成（即Fig1中左半部分，$N=6$）。每一层又分为两个子层（sub-layers）。第一个子层是多头自注意力机制（multi-head self-attention mechanism），第二个子层就是一个简单的全连接前馈网络。并且，每个子层我们都使用了[残差连接](http://shichaoxin.com/2022/01/07/论文阅读-Deep-Residual-Learning-for-Image-Recognition/)和[Layer Normalization](http://shichaoxin.com/2022/03/19/论文阅读-Layer-Normalization/)。也就是说，每个子层的输出为$LayerNorm(x+Sublayer(x))$。为了方便[残差连接](http://shichaoxin.com/2022/01/07/论文阅读-Deep-Residual-Learning-for-Image-Recognition/)，我们将每个子层以及embedding层的输出的维度都设置为$d_{model}=512$。

👉**Decoder:**

解码器也是由6个相同的层构成（即Fig1中右半部分，$N=6$）。每个层包含有三个子层，其中一个子层对编码器的输出执行多头注意力。和编码器一样，我们对解码器中的每个子层也都使用了[残差连接](http://shichaoxin.com/2022/01/07/论文阅读-Deep-Residual-Learning-for-Image-Recognition/)和[Layer Normalization](http://shichaoxin.com/2022/03/19/论文阅读-Layer-Normalization/)。解码器中有一个子层是Masked Multi-Head Attention，是基于Multi-Head Attention修改得来的，其作用是确保位置$i$的预测只能依赖于位置小于$i$的已知输出。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/4.png)

⚠️只有编码器的最后一层和解码器的每一层相连。

## 3.2.Attention

注意力功能（an attention function）可以描述为一个query和一组key-value对通过映射得到output，这里的query、keys、values和output都是向量（个人理解：句子中的每个单词都会有自己的query、key、value和output）。通过对values进行加权求和可得到output，而每一个value的权值则通过query和对应的key计算得到。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/2.png)

### 3.2.1.Scaled Dot-Product Attention

我们称我们这种特殊的注意力机制为Scaled Dot-Product Attention（见Fig2左）。输入包含queries（维度为$d_k$）、keys（维度为$d_k$）、values（维度为$d_v$）。首先计算某个query和所有keys的[点积（dot product）](http://shichaoxin.com/2019/08/27/数学基础-第七课-矩阵与向量/#64数量积)（即Fig2左中的MatMul），得到的值再除以$\sqrt{d_k}$（即Fig2左中的Scale），之后再通过softmax函数得到该key所对应的value的权值。

假设注意力机制的输入为$\mathbf{x}$（假设我们的句子只有两个单词），输出为$\mathbf{z}$：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/5.png)

首先我们需要将每个单词的[词嵌入向量](http://shichaoxin.com/2021/01/17/深度学习基础-第四十五课-自然语言处理与词嵌入/)（我们使用的[词嵌入向量](http://shichaoxin.com/2021/01/17/深度学习基础-第四十五课-自然语言处理与词嵌入/)的维度为$d_{model}=512$，详见第3.2.2部分）转化为我们需要的维度（即$d_k$或$d_v$，转化矩阵$W$的解释详见第3.2.2部分）：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/6.png)

则第一个单词的output的计算可表示为下图：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/7.png)

>个人理解：$\mathbf{z}_1 = 0.88 \mathbf{v}_1 + 0.12 \mathbf{v}_2$。所以是一个单词的输出结果会考虑到序列中的所有单词。计算第二个单词的输出时会重新计算各个value的权值。

我们把多个queries、keys和values分别打包成矩阵Q、K和V。Scaled Dot-Product Attention的输出可表示为：

$$Attention(Q,K,V)=softmax\left( \frac{QK^T}{\sqrt{d_k}} \right)V \tag{1}$$

用图可表示为：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/8.png)

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/9.png)

有两种常见的注意力：加法注意力（additive attention）和点积注意力（dot-product (multiplicative) attention）。在理论上，两种注意力的复杂度相近，但是点积注意力可以利用矩阵乘法，所以在实际操作中，点积注意力速度更快，空间效率（space-efficient）更高。因此，我们选择使用点积注意力。

但对于较小的$d_k$，两种注意力机制的表现差不多，但是对于较大的$d_k$，加法注意力优于点积注意力。我们怀疑是因为对于较大的$d_k$，点积的增长幅度较大，从而将softmax函数推到梯度非常小的区域，因此为了抵消这种影响，我们对点积注意力进行了Scale（即除以$\sqrt{d_k}$）。

### 3.2.2.Multi-Head Attention

我们发现了一种更好的做法，将queries、keys、values分别通过线性映射将其维度转换到$d_k,d_k,d_v$（queries、keys、values的原始维度为$d_{model}$）。线性映射一共执行$h$次，各次的映射可以并行化。然后通过Scaled Dot-Product Attention我们就可以得到$h$个维度为$d_v$的values，将这些values拼接在一起（Concat），再通过一次线性映射得到最终的输出。整个流程见Fig2右，我们将其称为多头注意力（Multi-Head Attention）。

多头注意力使模型能够共同关注来自不同位置的不同表征子空间（representation subspaces）的信息。单头注意力则会抑制这一特性。

$$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O$$

$$where \  head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)$$

>从公式可以看出，Multi-Head中的Head指的是并行做了多少次线性映射，即$h$。

$W$为参数矩阵：$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, W_i^K \in \mathbb{R}^{d_{model} \times d_k}, W_i^V \in \mathbb{R} ^{d_{model} \times d_v}, W^O \in \mathbb{R}^{hd_v \times d_{model}}$。

用图可表示为：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/10.png)

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/11.png)

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/12.png)

汇总到一张图：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/13.png)

>因为有[残差连接](http://shichaoxin.com/2022/01/07/论文阅读-Deep-Residual-Learning-for-Image-Recognition/)，所以子层的输入和输出维度应该保持一致。

在本文中，我们设置$h=8,d_k=d_v=d_{model}/h=64$。由于每个Head的维度降低，总的计算成本与全维度单头注意力的计算成本相似。

### 3.2.3.Applications of Attention in our Model

Transformer通过三种不同的方式使用多头注意力，见下图的1，2，3：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/14.png)

每个部分的解释见下：

1. 在“encoder-decoder attention”层中（个人理解：即编码器和解码器连接的部分，也就是解码器中间的那个多头注意力层，见Fig1），queries来自该解码器的上一层，keys和values来自编码器的输出。这样就使得解码器中的每个位置都能注意到输入序列的所有位置（个人理解：“keys和values来自编码器的输出”有3种可能的实现方式：1）编码器把最后一层的keys和values直接传给解码器的每一层；2）编码器只是把最后一层的转换矩阵$W^K,W^V$传给了解码器的每一层，解码器需要用上一个子层的输出自行计算keys和values；3）编码器把最后一层的输出$\mathbf{z}$传给了解码器的每一层，解码器需要自己学习转换矩阵$W^K,W^V$。因为本人未阅读源码，所以不确定是用的哪种方式，在此记录下自己的猜测）。
2. 编码器中包含的自注意力层（self-attention layers），其所有的keys，values和queries都来自上一层的输出。编码器中的每个位置都能注意到上一层的所有位置（个人理解：类似在第1点中提到的，“其所有的keys，values和queries都来自上一层的输出”也可以有3种方式，个人觉得应该是根据上一层的输出自行学习自己的转换矩阵）。
3. 该自注意力层是为了让每个位置注意到该位置以及之前所有的位置，而不关注该位置之后的位置。个人理解这个Masked Multi-Head Attention主要是在训练时起作用，因为测试阶段，单词是一个接一个的被预测出来的，不存在知道后续位置信息的情况。只有在训练阶段，我们已经知道了整个翻译好的GT，所以在模拟逐个单词被翻译出来的场景时，需要屏蔽掉后续位置上的单词。Mask的处理方式为将非法连接在Scaled Dot-Product Attention中对应的softmax输出置为$-\infty$：

	![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/15.png)
	
	![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/16.png)
	
后续又在网上查了一些关于Masked Multi-Head Attention的介绍，我比较赞同的一种说法是Masked Multi-Head Attention主要是为了支持训练的并行化。Transformer是一个auto-regressive的序列模型（见下图左，下图右为非auto-regressive的序列模型）：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/17.png)

auto-regressive的预测单词是一个接一个出来的，而非auto-regressive的预测单词是一起出来的。假如编码器的输入为“I love China”，GT为“我爱中国”，我们将GT作为解码器的输入，并行化分为5个分支：

1. 第一个分支的输入为“S”（句子起始标识），需屏蔽“我，爱，中，国”，然后计算第一个单词输出为“我”的loss。
2. 第二个分支的输入为“S，我”，需屏蔽“爱，中，国”，然后计算第二个单词输出“爱”的loss。
3. 第三个分支的输入为“S，我，爱”，需屏蔽“中，国”，然后计算第三个单词输出“中”的loss。
4. 第四个分支的输入为“S，我，爱，中”，需屏蔽“国”，然后计算第四个单词输出“国”的loss。
5. 第五个分支的输入为“S，我，爱，中，国”，没有需要屏蔽的内容，然后计算第五个单词输出“E”（句子结束标识）的loss。

然后计算这5个loss的总和作为最终的loss，这样的话可以大大减少训练时间。而对于测试阶段，其无法并行化，Masked Multi-Head Attention并没有实际作用，只是为了和训练结构保持一致。

通常称这种防止标签泄漏的mask为sequence mask。在NLP中还有另外一种常用的mask，称为padding mask，用于处理非定长序列（比如输入句子的长度通常都是不一样的，通过padding mask将其补齐为一样的长度，方便模型处理）。

## 3.3.Position-wise Feed-Forward Networks

编码器或解码器中每个层（block）除了注意力子层，还有一个全连接前馈网络（feed-forward network，FFN），这个网络分别作用于每个位置，如下图所示：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/5.png)

FFN其实只有一层（共有$d_{ff}=2048$个神经元），激活函数为ReLU函数：

$$FFN(x) = \max (0,xW_1+b_1)W_2+b_2 \tag{2}$$

各个位置上使用的FFN都是完全一样的，但FFN中各层之间的参数是不一样的（如上图所示，$z$到FFN的参数$W_1$和FFN到$r$的参数$W_2$是不一样的）。

## 3.4.Embeddings and Softmax

和其他序列转化模型类似，我们使用学习好的[词嵌入矩阵](http://shichaoxin.com/2021/01/17/深度学习基础-第四十五课-自然语言处理与词嵌入/#3嵌入矩阵)将input tokens和output tokens转化为$d_{model}$维的向量。我们还使用了softmax来预测下一个token的概率。在两个embedding layers（下图中1，2）和pre-softmax linear transformation中（下图中3）使用同一个权重矩阵。但在embedding layers中，我们将权重矩阵乘以$\sqrt{d_{model}}$（即之前提到的[词嵌入矩阵](http://shichaoxin.com/2021/01/17/深度学习基础-第四十五课-自然语言处理与词嵌入/#3嵌入矩阵)）。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/18.png)

## 3.5.Positional Encoding

因为我们的模型没有循环和卷积结构，为了给模型加入位置之间的序列信息，我们添加了“positional encodings”（见Fig1）。positional encodings的维度也是$d_{model}$，和词嵌入向量维度一样，这样两者可以相加（见Fig1）。

positional encodings的计算见下：

$$PE_{(pos,2i)}=\sin (pos / 10000^{2i/d_{model}})$$

$$PE_{(pos,2i+1)}=\cos (pos / 10000^{2i/d_{model}})$$

其中，$pos$是token在序列中的位置，$i$的范围是$[0,1,...,d_{model}/2]$。例如，句子中第一个词（$pos=1$）的PE（positional encodings）为：

$$PE_{pos} = PE_1 = [\sin(1/10000^{0/512}), \cos(1/10000^{0/512}), \sin(1/10000^{2/512}), \cos(1/10000^{2/512}),...]$$

在表3的E行，我们和其他positional encodings方法（Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.）进行了比较。

# 4.Why Self-Attention

这一部分我们比较了self-attention layers和recurrent、convolutional layers这三种方式各自将长度不定的symbol representation $(x_1,...,x_n)$转化成定长的$(z_1,...,z_n)$时的一些特点。有三点使得我们最终选择了self-attention。

第一点是每层的总计算复杂度。第二点是并行化的最小计算量。

第三点是网络中长期依赖关系之间的路径长度。在许多序列转换任务中，学习长期依赖关系是一个关键挑战。而信号在网络中穿过的路径长度是学习长期依赖关系的一个重要因素。路径越短，越容易学习长期依赖关系。因此，我们还比较了上述三种方式中任意两个输入和输出位置之间的最大路径长度。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/19.png)

对比结果见表1。就计算复杂度来说，如果$n<d$，则self-attention layers比recurrent快。为了提高较长序列的计算性能，self-attention可以限制为只关注以目标位置为中心，大小为$r$的邻域。但这也会导致最大路径长度的增加。我们计划在未来的工作中进一步研究这种方式。

此外，self-attention是一个更容易被解释的模型（即不像CNN是个黑盒模型）。

注意力的可视化：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/20.png)

Fig3为编码器第5层自注意力机制长期依赖关系的一个例子。可以看到单词“making”注意到了很多距离较远的单词，完成了词组“making ... more difficult”。

>并且从Fig3中可以看出序列使用了padding mask，使用标识`<pad>`将序列长度保持一致。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/21.png)

Fig4也是编码器第5层，上面为第5个头的注意力可视化，下面为单词“its”的第5，6个头的注意力可视化。

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/22.png)

不同的注意力头可以学到不同的句子结构，见Fig5，是来自编码器第5层两个不同的头的可视化结果。

# 5.Training

本部分介绍模型的训练细节。

## 5.1.Training Data and Batching

训练使用标准的WMT 2014 English-German数据集，共包含450万个句子对。数据集中的句子使用byte-pair encoding进行编码，单词表共有token（即单词）约37000个（其实是37000个英语-德语单词对）。对于英法互译，我们使用更大的WMT 2014 English-French数据集，包含3600万个句子对，单词表共有32000个token。长度相近的句子被放在同一个batch里。每个batch包含一组句子对，会用到大约25000个单词对。 

>byte-pair encoding是一种简单的数据压缩方法。这种方法用数据中不存在的一个字节表示最常出现的连续字节数据。比如我们要编码数据`aaabdaaabac`，字节对`aa`出现次数最多，所以我们用数据中没有出现的字节`Z`替换`aa`得到`ZabdZabac`，此时，字节`Za`出现的次数最多，我们用另一个字节`Y`来替换`Za`得到`YbdYbac`，同理，再用`X`替换`Yb`得到`XdXac`，由于不再有重复出现的字节对，所以这个数据不能再被进一步压缩。解压的时候，就是按照相反的顺序执行替换过程。

## 5.2.Hardware and Schedule

我们在一台配有8块NVIDIA P100 GPUs的机器上训练模型。对于base model（见表3），训练一个batch（原文用词为training step）大约花费0.4秒。我们共训练了100000个batch，总耗时12个小时。对于big model（见表3），训练一个batch大约花费1秒，我们共训练了300000个batch，总耗时3.5天。

## 5.3.Optimizer

使用[Adam优化算法](http://shichaoxin.com/2020/03/19/深度学习基础-第十九课-Adam优化算法/)，其中$\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$。学习率的变化遵循下式：

$$lrate = d_{model}^{-0.5} \cdot \min (step\_ num^{-0.5},step\_num \cdot warmup\_steps^{-1.5}) \tag{3}$$

其中，$warmup\\_steps=4000$。

## 5.4.Regularization

训练阶段使用了三种方式的正则化。

👉**Residual Dropout**

对每个子层的输出进行[dropout](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#5dropout正则化)。此外，对词嵌入向量和positional encodings的和也进行了[dropout](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#5dropout正则化)。对于base model（见表3），$P_{drop}=0.1$。

👉**Label Smoothing**

在训练阶段，我们还使用了[标签正则化](http://shichaoxin.com/2021/11/29/论文阅读-Rethinking-the-Inception-Architecture-for-Computer-Vision/#7model-regularization-via-label-smoothing)，并且设$\epsilon_{ls}=0.1$。这一操作提高了模型的准确性和[BLEU分数](http://shichaoxin.com/2021/03/03/深度学习基础-第四十七课-BLEU得分/)。

# 6.Results

## 6.1.Machine Translation

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/23.png)

在WMT 2014 English-to-German翻译任务中，我们big model（见表3）的[BLEU分数](http://shichaoxin.com/2021/03/03/深度学习基础-第四十七课-BLEU得分/)比之前最好成绩还高2.0，达到了SOTA的28.4。在8块P100 GPUs上训练了3.5天。即使是我们的base model也比之前最好的成绩要高，并且其训练成本是所有模型中最低的。

在WMT 2014 English-to-French翻译任务中，我们big model的[BLEU分数](http://shichaoxin.com/2021/03/03/深度学习基础-第四十七课-BLEU得分/)为41.0，优于之前所有方法，并且计算成本不到之前方法的四分之一。这里我们使用$P_{drop}=0.1$，而不是表3中的0.3。

我们还使用了[Beam Search](http://shichaoxin.com/2021/02/23/深度学习基础-第四十六课-Beam-Search/)，其中beam size=4，length penalty $\alpha$=0.6。这些超参数是通过在验证集（the development set）上实验得到的。我们设置输出序列的最大长度为输入长度+50。

## 6.2.Model Variations

为了评估Transformer模型中不同组件的重要程度，我们对base model进行了多种方式的改造，并在English-to-German翻译任务的development set，newstest2013数据集上进行性能度量。这里同样也使用了[Beam Search](http://shichaoxin.com/2021/02/23/深度学习基础-第四十六课-Beam-Search/)，参数设置和第6.1部分相同。实验结果见表3：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/24.png)

表3中的A行，我们改变了头的个数以及key和value的维度。可以看出，头太多或者太少都会导致[BLEU分数](http://shichaoxin.com/2021/03/03/深度学习基础-第四十七课-BLEU得分/)的下降。

表3中的B行，我们发现key的维度的下降会导致[BLEU分数](http://shichaoxin.com/2021/03/03/深度学习基础-第四十七课-BLEU得分/)的下降。从表3中的C行和D行，我们发现模型越大效果越好，并且[dropout](http://shichaoxin.com/2020/02/01/深度学习基础-第十一课-正则化/#5dropout正则化)的使用很好的避免了过拟合。表3中E行的解释见第3.5部分。

## 6.3.English Constituency Parsing

为了评估Transformer是否可以推广到其他任务，我们对English Constituency Parsing任务进行了实验，本部分不再详述，结果见表4：

![](https://github.com/x-jeff/BlogImage/raw/master/AIPapers/Transformer/25.png)

结论就是Transformer在English Constituency Parsing任务上表现也很优异。

# 7.Conclusion

本文中，我们提出了Transformer，第一个完全基于注意力机制的序列转化模型，并没有使用编码-解码框架中最常见的循环层。

相比[RNN](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)和[CNN](http://shichaoxin.com/2020/07/04/深度学习基础-第二十八课-卷积神经网络基础/)，Transformer训练更快，效果更好。此外，Transformer可以很好的扩展到其他类型的任务中。

源码地址：[https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)。

# 8.原文链接

👽[Attention Is All You Need](https://github.com/x-jeff/AI_Papers/blob/master/Attention%20Is%20All%20You%20Need.pdf)

# 9.参考资料

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
2. [Illustrated Guide to Transformers- Step by Step Explanation](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)
3. [在测试或者预测时，Transformer里decoder为什么还需要seq mask？](https://www.zhihu.com/question/369075515)
4. [Transformer相关——（7）Mask机制](https://ifwind.github.io/2021/08/17/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%887%EF%BC%89Mask%E6%9C%BA%E5%88%B6/)
5. [对Transformer中的Positional Encoding一点解释和理解](https://zhuanlan.zhihu.com/p/98641990)
6. [字节对编码（wiki百科）](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%AF%B9%E7%BC%96%E7%A0%81)