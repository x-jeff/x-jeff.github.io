---
layout:     post
title:      【从零开始构建大语言模型】【1】【Understanding large language models】
subtitle:   What is an LLM?，Applications of LLMs，Stages of building and using LLMs，Introducing the transformer architecture，Utilizing large datasets，A closer look at the GPT architecture，Building a large language model
date:       2025-02-18
author:     x-jeff
header-img: blogimg/20221107.jpg
catalog: true
tags:
    - Large Language Models
---
>【从零开始构建大语言模型】系列博客为"Build a Large Language Model (From Scratch)"一书的个人读书笔记。
>
>* 原书链接：[Build a Large Language Model (From Scratch)](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167?crid=228R4JI0P0QFR&dib=eyJ2IjoiMSJ9.XvZyIer9iV133BWXqNiVt_OOJXZheO54dvZtQly8MC25PNYZrN3OWsGLjbg3I0G9hI3LkjwhsORxvHIob3nvCZFgdSSQEFe07VkehijGxT03n4Amdw7lnXxnsOUuWXeglfHnewCcV3DjL9zWHELfh5DG1ZErzFym3S6ZxSuFzNvoPkaq0uDlD_CKwqHdC0KM_RdvIqF0_2RudgvzRli0V155KkusHRck3pG7ybp5VyqKDC_GgL_MEywLwLhFgX6kOCgV6Rq90eTgSHFd6ac8krpIYjsHWe6H3IXbfKGvMXc.473O1-iUZC0z2hdx8L5Z5ZTNxtNV9gNPw_mE7QZ5Y90&dib_tag=se&keywords=raschka&qid=1730250834&sprefix=raschk,aps,162&sr=8-1&linkCode=sl1&tag=rasbt03-20&linkId=84ee23afbd12067e4098443718842dac&language=en_US&ref_=as_li_ss_tl)。
>* 官方示例代码：[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
>
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.What is an LLM?

large language model中的large不仅仅指模型的参数规模，还指其训练所使用的庞大数据集。

![Fig1.1](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/1.png)

# 2.Applications of LLMs

![Fig1.2](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/2.png)

Fig1.2是一个文本生成的应用例子。

# 3.Stages of building and using LLMs

大多数的LLM都是基于PyTorch实现的。

研究表明，在模型性能方面，定制化的LLM（针对特定任务或领域优化的模型）往往优于通用LLM（比如ChatGPT），后者旨在适用于更广泛的应用场景。

定制化LLM的优势：

1. 保护数据隐私，可以不将敏感数据共享给OpenAI等第三方LLM提供商。
2. 可以开发更小型的LLM，部署在用户设备上，比如笔记本电脑或智能手机。
3. 可以根据需求自由控制模型的更新和调整。

构建LLM的一般流程包括pre-train和fine-tune，如Fig1.3所示。

![Fig1.3](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/3.png)

# 4.Introducing the transformer architecture

大多数的LLM都基于[transformer](https://shichaoxin.com/2022/03/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Attention-Is-All-You-Need/)架构，原始的[transformer](https://shichaoxin.com/2022/03/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Attention-Is-All-You-Need/)架构最初是用于机器翻译的，其简化示意图见Fig1.4。

![Fig1.4](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/4.png)

>对于transformer的详细讲解，请移步另一篇博客：[【论文阅读】Attention Is All You Need](https://shichaoxin.com/2022/03/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Attention-Is-All-You-Need/)。

基于[transformer](https://shichaoxin.com/2022/03/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Attention-Is-All-You-Need/)架构的两个变体：

![Fig1.5](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/5.png)

>* 对于BERT的详细讲解，请移步博客：[【论文阅读】BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding](https://shichaoxin.com/2024/08/12/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/)。
>* 对于GPT系列的详细讲解，请移步博客：[【LLM】一文读懂ChatGPT背后的技术](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/)。

[BERT](https://shichaoxin.com/2024/08/12/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/)专注于掩码词预测，而[GPT](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/)专注于文本生成。

[GPT](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/)系列模型擅长zero-shot learning和few-shot learning，如Fig1.6所示。

* zero-shot learning：指模型能够在完全未见过的任务上进行泛化，无需任何特定示例。
* few-shot learning：指模型能够从用户提供的极少量示例中学习，然后执行相应任务。

![Fig1.6](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/6.png)

# 5.Utilizing large datasets

表1.1列出了GPT-3预训练所使用的数据集。

![table1.1](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/7.png)

表1.1中的token指的是模型读取的最小文本单位。在数据集中，token的数量大致等同于文本中的单词和标点符号总数。

预训练LLM需要大量计算资源，成本极高，比如，GPT-3的预训练成本约为460万美元。

# 6.A closer look at the GPT architecture

[GPT](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/)模型的预训练任务相对简单，仅基于对下一个单词的预测，如Fig1.7所示。

![Fig1.7](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/8.png)

对下一个单词进行预测的任务属于是一种自监督学习（self-supervised learning），本质上是一种自标注（self-labeling）方法。这意味着我们不需要对训练数据进行标注，而是可以利用数据本身的结构：将句子或文档中的下一个单词作为模型要预测的标签。因此可以使用海量的无标签文本数据集来训练LLM。

[GPT](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/)的整体架构仅保留了[transformer](https://shichaoxin.com/2022/03/26/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-Attention-Is-All-You-Need/)的解码器，去除了编码器，如Fig1.8所示。这些只用解码器的模型（decoder-style models），比如[GPT](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/)，是逐词进行预测的，因此，这些模型也被称为**自回归模型（autoregressive）**。自回归模型会将之前的输出作为输入，用于未来的预测。这种机制增强了生成文本的连贯性。

![Fig1.8](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/9.png)

模型能够执行未经过专门训练的任务，这种现象被称为**涌现行为（emergent behavior）**。这种能力并非通过显式训练获得，而是由于模型在训练过程中接触到了大量多语言数据，并在不同的语境下学习了语言模式，从而自然地表现出这一能力。[GPT](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/)模型能够“学习”不同语言之间的翻译模式，并执行翻译任务，即使它们并未专门为此训练，这一点凸显了大型生成式语言模型的强大能力和优势。这意味着，我们可以使用单一模型来完成多种任务，而无需为每项任务训练独立的模型。

# 7.Building a large language model

我们将按照Fig1.9的步骤构建一个LLM。

![Fig1.9](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/1/10.png)

一共分为3个阶段：

* 构建LLM
* pre-train
* fine-tune
