---
layout:     post
title:      【论文阅读】BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding
subtitle:   BERT
date:       2024-08-12
author:     x-jeff
header-img: blogimg/20210721.jpg
catalog: true
tags:
    - Natural Language Processing
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

对语言模型进行预训练已经被证实对提升许多NLP任务是有效的。

将预训练语言表征应用到下游任务通常有两种策略：feature-based和fine-tuning。feature-based策略的代表方法是ELMo，其针对每一个下游任务，构造一个与这个任务相关的神经网络（实际使用的是[RNN框架](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)），然后将预训练好的表征作为额外的特征和原有输入一起喂给模型。fine-tuning策略的代表方法是[GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)，其将预训练好的模型应用在下游任务上，且不需要做过多的修改，预训练好的参数会在下游数据上进行fine tune。这两种方法在预训练阶段都是使用相同的目标函数，且都是单向的语言模型（个人注解：语言模型通常就是单向的，比如给定一些词，预测下一个词是什么）。

>ELMo：Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018a. Deep contextualized word representations. In NAACL.
>
>个人注解：BERT和ELMo都是动画片芝麻街里的人物名，这也开启了NLP芝麻街系列，坐等后续会不会有新的芝麻街人物出现😂。
>
>![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/1.png)

我们认为当前的技术限制了预训练表征的能力，特别是对于fine-tuning的方法。主要的局限性在于标准的语言模型是单向的，这限制了在预训练期间对于可使用框架的选择。比如，在[OpenAI的GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)中，作者使用了一种从左到右的框架，即每个token只能关注到前面的token。但这对于sentence-level的任务来说是次优的，比如根据一个句子判断情绪，无论是从左到右分析这个句子，还是从右到左分析这个句子，得到的结果应该都是一样的。甚至对于一些token-level的任务也不是最优的，比如Q&A任务，我们可以看完整个问题再去选答案，并不需要一个接一个的预测下一个词。因此，我们认为如果把两个方向的信息都放进来的话，是可以提升这些任务的性能的。

本文中，我们完善了基于fine-tuning的方法，提出了BERT：**B**idirectional **E**ncoder **R**epresentations from **T**ransformers。BERT通过使用一个带掩码的语言模型（masked language model，MLM）缓解了语言模型的单向限制。MLM随机屏蔽输入中的一些token，目的是仅根据其上下文预测出被屏蔽的token（个人注解：类似完形填空）。与从左到右语言模型预训练不同，MLM能够融合左右上下文，这使得我们可以预训练一个深的双向[Transformer](http://shichaoxin.com/2022/03/26/论文阅读-Attention-Is-All-You-Need/)。除了MLM外，我们还训练了另外一个任务，叫做“下一个句子的预测”（“next sentence prediction”），其核心思想是给定两个句子，让模型去判断这两个句子在原文中是不是相邻的，这能使模型学习到一些句子层面的信息。我们的贡献主要有以下3点：

1. 我们证明了双向预训练对语言表征的重要性。
2. 我们证明了，一个好的预训练模型，就不需要再对特定的任务做一些特定的模型改动了。在基于fine-tuning的方法中，BERT是第一个在一系列NLP任务（包括sentence-level和token-level）上达到SOTA的。
3. 开源代码和预训练模型：[https://github.com/google-research/bert](https://github.com/google-research/bert)。

>个人注解：作者主要介绍了两个之前的研究，一个是ELMo，其是双向+[RNN框架](http://shichaoxin.com/2020/11/22/深度学习基础-第四十课-循环神经网络/)，另一个是[GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)，其是单向+[Transformer框架](http://shichaoxin.com/2022/03/26/论文阅读-Attention-Is-All-You-Need/)。而BERT就是结合了上述两种思想，是双向+[Transformer框架](http://shichaoxin.com/2022/03/26/论文阅读-Attention-Is-All-You-Need/)。

# 2.Related Work

不再详述。

# 3.BERT

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/2.png)

BERT有两个步骤：预训练和fine-tuning。预训练是在一个没有标签的数据集上进行的。fine-tuning是在下游有标签的数据上进行的。每个下游任务都有特定的fine-tuned模型，即使它们都是用相同的预训练参数初始化的。

BERT的一个显著特征就是不同任务使用统一的模型框架。预训练模型框架和最终下游任务的模型框架之间的差异很小。

👉**Model Architecture**

BERT的模型框架是一个多层双向Transformer编码器。

我们将层数（即Transformer blocks）记为$L$，hidden size（即隐藏层大小）记为$H$，自注意力头的数量记为$A$。我们有两个模型：$\text{BERT}_{\text{BASE}}$（$L=12,H=768,A=12$，总参数量为110M）和$\text{BERT}_{\text{LARGE}}$（$L=24,H=1024,A=16$，总参数量为340M）。

$\text{BERT}_{\text{BASE}}$和[GPT1](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/#1gpt1)的参数量相当。

👉**Input/Output Representations**

为了使BERT可以处理各种下游任务，BERT的输入是一个序列，其即可以是一个句子，也可以是一个句子对（比如`<Question,Answer>`）。这里的句子指的是一段连续的文字，并不一定真的只是一句话。

我们用的切词方法是WordPiece。假设我们按照空格切词的话，一个词作为一个token，我们的数据量比较大，从而导致词典大小也特别大，可能会达到百万级别，因此，按照WordPiece的处理方法，如果一个词出现的概率不大的话，我们可以把它切开，看它的一个子序列（可能是一个词根），若子序列出现的概率比较大的话，我们就只保留这个子序列就可以了。这样我们可以把一个相对比较长的词，切成多个片段，这些片段是经常出现的，这样我们就可以用一个相对较小的词典（本文中，是一个30,000 token的词典）来表示一个较大的文本了。每个序列的第一个token是一个特殊的分类token：`[CLS]`（即classification）。`[CLS]`最后对应的输出代表的是整个序列的一个信息。如果是句子对作为一个序列，则需要对这两个句子进行区分，我们有两种方法。第一个方法是在句子后面放一个特殊的词：`[SEP]`（即separate）。第二个方法是学习一个嵌入层来表示这个句子是句子$A$还是句子$B$。如Fig1所示。我们将input embedding记为$E$，`[CLS]`对应的最终隐藏向量（final hidden vector）为$C \in \mathbb{R}^H$，第$i$个输入token的最终隐藏向量为$T_i \in \mathbb{R}^H$。

>WordPiece：Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural ma- chine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/3.png)

如Fig2所示，将词转化为BERT的input embedding包含3部分，第一部分是词本身的embedding，第二部分是词在哪个句子的embedding，第三部分是位置的embedding（在[Transformer](http://shichaoxin.com/2022/03/26/论文阅读-Attention-Is-All-You-Need/)中，位置信息是手动构造出来的一个矩阵，而在BERT中，第二部分和第三部分都是通过学习得来的）。

## 3.1.Pre-training BERT

BERT的预训练使用了2种非监督任务。

👉**Task #1: Masked LM**

为了训练深层双向表征，我们随机mask掉了输入中的一些词，然后来预测这些被mask的词。我们将这一过程称为“masked LM”（MLM）。序列中的每个词（除了特殊的token，比如`[CLS]`和`[SEP]`）都有15%的概率被mask掉。

被mask掉的词会使用一个特殊的token：`[MASK]`来代替。这会存在一个问题，在fine-tune的时候，我们不用mask，所以对于预训练和fine-tune，喂给模型的数据可能会稍稍有些不一样。为了缓解这个问题，对于这15%被选中的词：1）80%被真正的mask掉，替换为`[MASK]`；2）10%被替换为一个随机的token；3）10%不做任何变化。相关的消融实验见附录C.2。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/4.png)

👉**Task #2: Next Sentence Prediction (NSP)**

很多重要的下游任务，比如QA和自然语言推理，都是基于理解两句话之间的关系，而语言建模并不能直接捕捉到这种关系。为了训练一个可以理解句子关系的模型，我们预训练了一个二值化的下一个句子预测（next sentence prediction，NSP）任务，该任务可以从任何语料库中轻松生成。具体来说，对于每个预训练样本，有句子$A$和句子$B$，有50%的概率句子$B$在原始语料库中就是跟在句子$A$后面的（标签为`IsNext`），还有50%的概率句子$B$是随机选的，原本就不在句子$A$后面（标签为`NotNext`）。在Fig1中，$C$就是用来预测这个标签的。尽管这个策略很简单，但却有效。最终模型在NSP上达到了97%-98%的准确率。如果不进行fine-tune，向量$C$就不是一个有意义的句子表征，因为它是用NSP训练的。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/5.png)

👉**Pre-training data**

预训练使用了两个数据集：BooksCorpus（包含800M个词）和English Wikipedia（包含2,500M个词）。

## 3.2.Fine-tuning BERT

这里通过几个BERT fine-tune的例子来理解这一过程。

如下图所示，第一个例子的下游任务是句子分类，输入是一个句子，输出是一个类别。比如输入句子“This is good”，输出这个句子所表达的情绪是正面的还是负面的。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/6.png)

第二个例子中，输入是一个序列，输出是同样长度的另外一个序列。比如输入是一个句子，输出是对句子中每个词词性的分类。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/7.png)

第三个例子中，输入是两个句子，输出是一个类别。比如NLI（Natural Language Inference）任务，主要用于判断两句话之间的逻辑关系。具体来说，给定一对句子，称为前提（premise）和假设（hypothesis），NLI的任务是确定假设相对于前提的关系，可以是以下三种之一：1）蕴含（entailment）：假设能够从前提中推导出来；2）矛盾（contradiction）：假设与前提相矛盾；3）中立（neutral）：假设与前提既不蕴含也不矛盾，可能提供了与前提无关的新信息。NLI任务示意：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/8.png)

fine-tune示意：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/9.png)

与预训练比，fine-tune相对便宜一些。所有的结果只需要用TPU跑一个小时，或者使用GPU跑几个小时也可以。

# 4.Experiments

本部分展示了BERT在11个NLP任务上的fine-tune结果。

## 4.1.GLUE

GLUE（General Language Understanding Evaluation） benchmark是多个NLP任务的集合。GLUE数据集的详细介绍见附录B.1。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/10.png)

## 4.2.SQuAD v1.1

SQuAD v1.1（Stanford Question Answering Dataset）是一个QA数据集。如下图所示，这里的答案是问题文本中的某个子序列，所以我们只要输出这个子序列开始的序号s和结束的序号e就可以了。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/11.png)

BERT fine-tune的过程可表示为：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/12.png)

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/13.png)

fine-tune后的结果：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/14.png)

## 4.3.SQuAD v2.0

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/15.png)

## 4.4.SWAG

SWAG（Situations With Adversarial Generations）数据集包含113k个句子对，该数据集用于判断句子之间的关系。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/16.png)

# 5.Ablation Studies

## 5.1.Effect of Pre-training Tasks

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/17.png)

## 5.2.Effect of Model Size

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/18.png)

## 5.3.Feature-based Approach with BERT

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/19.png)

# 6.Conclusion

最近一些实验表明使用非监督的预训练是非常好的。这使得训练样本不多的任务也可以使用深层单向框架。我们的主要贡献是将其进一步推广到深层双向框架，使同样的预训练模型能够处理广泛的NLP任务。

# 7.Appendix

主要分为3部分：

* BERT的额外实现细节见附录A。
* 额外的实验细节见附录B。
* 额外的消融实验见附录C。

## 7.A.Additional Details for BERT

### 7.A.1.Illustration of the Pre-training Tasks

👉**Masked LM and the Masking Procedure**

假设有未标注的句子“my dog is hairy”，我们选择第4个词hairy进行mask，则：

* 有80%的概率被真的mask，替换为`[MASK]`：“my dog is [MASK]”。
* 有10%的概率被替换为任意一个词：“my dog is apple”。
* 有10%的概率不做任何改变：“my dog is hairy”。

👉**Next Sentence Prediction**

NSP任务示例：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/20.png)

第二个例子中，单词`flightless`被WordPiece分成了两个词：`flight`和`less`，`##less`表示原本是和上一个词组合在一起的。

### 7.A.2.Pre-training Procedure

为了生成每个训练输入序列，我们从语料库中采样得到两段文字，视为我们定义的句子，即句子A和句子B。对于NSP任务，有50%的概率句子B真的是在句子A后面，还有50%的概率句子B是随机选择的。句子A和句子B的组合长度小于等于512个token。对于MLM任务，mask rate为15%。

训练用的batch size为256个序列（256个序列\*512个token=128,000 tokens/batch），共训练了1,000,000步，近似于在3.3 billion的语料库上训练了40个epoch。使用Adam，学习率为$1e-4$，$\beta_1=0.9$，$\beta_2=0.999$，L2 weight decay=0.01，前10,000步用于学习率warm up，学习率使用线性衰减。所有层的dropout概率都是0.1。使用[GELU](http://shichaoxin.com/2022/04/09/论文阅读-GAUSSIAN-ERROR-LINEAR-UNITS-(GELUS)/)激活函数。训练loss是平均MLM似然和平均NSP似然之和。

训练$\text{BERT}_{\text{BASE}}$使用了4块TPU（共16个TPU芯片）。训练$\text{BERT}_{\text{LARGE}}$使用了16块TPU（共64个TPU芯片）。每个模型都预训练了4天时间。

为了加速预训练，前90%步使用长度为128的序列，后10%步使用长度为512的序列。

### 7.A.3.Fine-tuning Procedure

对于fine-tune，除了batch size、学习率和训练的epoch数，剩余超参数和预训练是一样的。最优的超参数是和任务相关的，但我们发现如下一些超参数取值对所有任务来说效果都还可以：

* **Batch size**：16、32。
* **学习率（Adam）**：$5e-5$、$3e-5$、$2e-5$。
* **epoch数量**：2、3、4。

我们还发现，大型数据集（例如，100k+带标注的训练数据）对超参数选择的敏感度远低于小型数据集。fine-tune通常非常快。

### 7.A.4.Comparison of BERT, ELMo ,and OpenAI GPT

对于最近流行的表征学习模型：ELMo、[OpenAI GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)和BERT，我们研究了其差异。模型框架的比较见Fig3。BERT和[OpenAI GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)是fine-tuning方法，而ELMo是feature-based方法。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/21.png)

与BERT预训练方法最相似的是[OpenAI GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)，它在大型文本库上训练从左到右的Transformer LM。BERT和[GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)的训练方式有以下一些差异：

* [GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)在BooksCorpus（800M个词）上训练；BERT在BooksCorpus（800M个词）和Wikipedia（2,500M个词）上训练。
* [GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)只在fine-tune阶段引入`[SEP]`和`[CLS]`；而BERT是在预训练阶段。
* [GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)训练了1M步，batch size为32,000个词；BERT也是训练了1M步，但batch size为128,000个词。
* [GPT](http://shichaoxin.com/2024/03/20/LLM-一文读懂ChatGPT背后的技术/)在所有fine-tune实验中都使用同样的学习率$5e-5$；而BERT根据fine-tune的任务选择特定的学习率。

第5.1部分的实验结果表明，BERT大部分的改进来自两个预训练任务以及其双向性。

### 7.A.5.Illustrations of Fine-tuning on Different Tasks

在不同任务上fine-tune BERT的示意见Fig4。任务特定的模型是基于BERT添加一个额外的输出层而形成的。(a)和(b)是sequence-level的任务，而(c)和(d)是token-level的任务。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/22.png)

## 7.B.Detailed Experimental Setup

### 7.B.1.Detailed Descriptions for the GLUE Benchmark Experiments.

GLUE benchmark包含以下数据集：

1. **MNLI**：Multi-Genre Natural Language Inference。
2. **QQP**：Quora Question Pairs。
3. **QNLI**：Question Natural Language Inference。
4. **SST-2**：Stanford Sentiment Treebank。
5. **CoLA**：Corpus of Linguistic Acceptability。
6. **STS-B**：Semantic Textual Similarity Benchmark。
7. **MRPC**：Microsoft Research Paraphrase Corpus。
8. **RTE**：Recognizing Textual Entailment。
9. **WNLI**：Winograd NLI。

## 7.C.Additional Ablation Studies

### 7.C.1.Effect of Number of Training Steps

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/23.png)

### 7.C.2.Ablation for Different Masking Procedures

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/AIPapers/BERT/24.png)

# 8.原文链接

👽[BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding](https://github.com/x-jeff/AI_Papers/blob/master/2024/BERT：Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.pdf)

# 9.参考资料

1. [BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ/?spm_id_from=333.880.my_history.page.click&vd_source=896374db59ca8f208a0bb9f453a24c25)
2. [【機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介](https://www.youtube.com/watch?v=gh0hewYkjgo)