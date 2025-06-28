---
layout:     post
title:      【LLM】LangChain：Chat with Your Data
subtitle:   用LangChain构建基于文档的智能问答系统
date:       2025-06-26
author:     x-jeff
header-img: blogimg/20220821.jpg
catalog: true
tags:
    - Large Language Models
---
>本文为参考DeepLearning.AI的"LangChain: Chat with Your Data"课程所作的个人笔记。
>
>课程地址：[https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)。
>
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

本文主要聚焦于如何使用LLM基于自己的文档进行问答对话。这部分的内容之前已经有过简单的讲解，请见：[Question and Answer](https://shichaoxin.com/2025/06/06/LLM-LangChain-for-LLM-Application-Development/)，本文将继续深入探讨。

# 2.Document Loading

LangChain支持超过80种不同的文档加载器。这些数据的来源可以是网页、数据库、YouTube、arXiv等，数据的格式可以是PDF、HTML、JSON、Word、PPT等。文档加载器的作用就是将这些数据加载到一个标准的文档对象中。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/1.png)

在[这里](https://shichaoxin.com/2025/06/06/LLM-LangChain-for-LLM-Application-Development/#5question-and-answer)我们介绍了CSV数据的加载，现在我们来看下PDF数据的加载：

````python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
print(len(pages))
````

打印PDF的页数：

````
22
````

打印首页的前500个单词：

````python
page = pages[0]
print(page.page_content[0:500])
````

````
MachineLearning-Lecture01  
Instructor (Andrew Ng):  Okay. Good morning. Welcome to CS229, the machine 
learning class. So what I wanna do today is ju st spend a little time going over the logistics 
of the class, and then we'll start to  talk a bit about machine learning.  
By way of introduction, my name's  Andrew Ng and I'll be instru ctor for this class. And so 
I personally work in machine learning, and I' ve worked on it for about 15 years now, and 
I actually think that machine learning i
````

查看首页的元数据，包含数据来源和页码：

````python
print(page.metadata)
````

````
{'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 0}
````

接下来我们再来看下YouTube数据的加载：

````python
#GenericLoader是一个通用的文档加载器，用于将不同的数据源与特定的解析器配对，以加载并解析文档内容
#FileSystemBlobLoader是一个blob数据加载器，用于从本地文件系统读取文件，并将其作为blob（即二进制文件对象）提供给GenericLoader使用
from langchain.document_loaders.generic import GenericLoader,  FileSystemBlobLoader
#OpenAIWhisperParser是一个基于OpenAI Whisper模型的音频解析器，将音频文件转为文本内容
from langchain.document_loaders.parsers import OpenAIWhisperParser
#YoutubeAudioLoader是一种特殊的blob加载器，它会：
#   1.下载指定YouTube视频的音频
#   2.将音频文件转为blob对象
#   3.用于后续由解析器（比如OpenAIWhisperParser）处理成文本
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    #YoutubeAudioLoader([url],save_dir),  # fetch from youtube
    FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
    OpenAIWhisperParser()
)
docs = loader.load()

print(docs[0].page_content[0:500])
````

````
"Welcome to CS229 Machine Learning. Uh, some of you know that this is a class that's taught at Stanford for a long time. And this is often the class that, um, I most look forward to teaching each year because this is where we've helped, I think, several generations of Stanford students become experts in machine learning, got- built many of their products and services and startups that I'm sure, many of you or probably all of you are using, uh, uh, today. Um, so what I want to do today was spend s"
````

下面是一个从网络链接加载数据的例子：

````python
#WebBaseLoader是一个网页加载器，可以从给定的URL中提取主要的网页正文内容，并将其包装为Document对象供后续使用
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
docs = loader.load()
print(docs[0].page_content[:500])
````

````















































































handbook/titles-for-programmers.md at master · basecamp/handbook · GitHub















































Skip to content














Navigation Menu

Toggle navigation




 













            Sign in
          


 


Appearance settings











        Product
        














            GitHub Copilot
          
        Write better code with AI
      








            GitHub Mod
````

可以看到，输出了很多空白，需要我们进一步后处理才能使用。接下来来看一个从Notion加载数据的例子：

````python
#NotionDirectoryLoader会解析从Notion导出的markdown格式的文件，将每个页面作为一个Document加载，供后续问答、摘要、搜索等任务使用
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()

print(docs[0].page_content[0:200])
````

````
# Blendle's Employee Handbook

This is a living document with everything we've learned working with people while running a startup. And, of course, we continue to learn. Therefore it's a document that
````

````python
print(docs[0].metadata)
````

````
{'source': "docs/Notion_DB/Blendle's Employee Handbook e367aa77e225482c849111687e114a56.md"}
````

# 3.Document Splitting

文档加载完成后，对于LLM来说，文档依旧过于庞大，因此我们需要对文档进行拆分：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/2.png)

LangChain支持多种不同的文本分割器：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/3.png)

这里我们看下`RecursiveCharacterTextSplitter`和`CharacterTextSplitter`的例子：

````python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size =26
chunk_overlap = 4
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
````

`chunk_size`和`chunk_overlap`的含义：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/4.png)

注意：`chunk_size`是分割文本片段的最大长度限制，并不要求分割得到的文本片段长度刚好为`chunk_size`。`chunk_size`默认是按字符数来计算文本长度的。

````python
text1 = 'abcdefghijklmnopqrstuvwxyz'
r_splitter.split_text(text1)
# 输出为：
# ['abcdefghijklmnopqrstuvwxyz']

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
r_splitter.split_text(text2)
# 输出为：
# ['abcdefghijklmnopqrstuvwxyz', 'wxyzabcdefg']

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
r_splitter.split_text(text3)
# 输出为：
# ['a b c d e f g h i j k l m', 'l m n o p q r s t u v w x', 'w x y z']
c_splitter.split_text(text3)
# 输出为：
# ['a b c d e f g h i j k l m n o p q r s t u v w x y z']
````

`RecursiveCharacterTextSplitter`是LangChain中最智能、最常用的文本分割器之一，专为保留语义完整性而设计，其会逐级递归尝试多个分隔符，以生成长度合适、语义连贯的文本片段（chunk）。

`CharacterTextSplitter`会用指定的分隔符对文本进行拆分，如果没有指定分隔符，那么默认分隔符为`"\n\n"`。有一种例外情况需要注意，如果文本片段的长度大于`chunk_size`，但这个文本片段中又找不到指定的分隔符，从而无法对其进行进一步的分割，此时就会保留该文本片段，不再继续分割，上述例子中的`c_splitter.split_text(text3)`就是这种情况。

````python
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator = ' ' #指定分隔符
)
c_splitter.split_text(text3)
# 输出为：
# ['a b c d e f g h i j k l m', 'l m n o p q r s t u v w x', 'w x y z']

text4 = "a b c d e f g h i j klmnopqrs t u v w x y z"
c_splitter.split_text(text4)
# 输出为：
# ['a b c d e f g h i j', 'i j klmnopqrs t u v w x y', 'x y z']
````

````python
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""
len(some_text) #输出为（字符数）：496

c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""] #先尝试用"\n\n"分割，如果不满足要求，再用"\n"分割，如果还不满足要求，再用" "分割，如果还是不满足要求，再用""分割（即逐字符分割）
)

c_splitter.split_text(some_text)
# 输出为：
# ['When writing documents, writers will use document structure to group content. This can convey to the reader, which idea\'s are related. For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns are the "backslash n" you see embedded in this string. Sentences have a period at the end, but also,',
#  'have a space.and words are separated by space.']

r_splitter.split_text(some_text) #相比c_splitter，r_splitter的分割更有逻辑
# 输出为：
# ["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related. For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.",
#  'Paragraphs are often delimited with a carriage return or two carriage returns. Carriage returns are the "backslash n" you see embedded in this string. Sentences have a period at the end, but also, have a space.and words are separated by space.']

#尝试分成更小的文本片段
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""]
)
r_splitter.split_text(some_text)
# 输出为：
# ["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related",
#  '. For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.',
#  'Paragraphs are often delimited with a carriage return or two carriage returns',
#  '. Carriage returns are the "backslash n" you see embedded in this string',
#  '. Sentences have a period at the end, but also, have a space.and words are separated by space.']

#句号被误放在了文本片段的开头，使用更复杂的正则表达式进行纠正
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
r_splitter.split_text(some_text)
# 输出为：
# ["When writing documents, writers will use document structure to group content. This can convey to the reader, which idea's are related.",
#  'For example, closely related ideas are in sentances. Similar ideas are in paragraphs. Paragraphs form a document.',
#  'Paragraphs are often delimited with a carriage return or two carriage returns.',
#  'Carriage returns are the "backslash n" you see embedded in this string.',
#  'Sentences have a period at the end, but also, have a space.and words are separated by space.']
````

````python
from langchain.document_loaders import PyPDFLoader
#PyPDFLoader返回页面列表，每一页是一个Document对象
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len #默认文本长度的计算方式为len，即按字符数计算
)
docs = text_splitter.split_documents(pages)

len(docs) #分割后的Document对象数量，输出为77
len(pages) #分割前的Document对象数量，即原始PDF的页数，输出为22
````

````python
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
notion_db = loader.load()

docs = text_splitter.split_documents(notion_db)

len(notion_db) #输出为：52
len(docs) #输出为：353
````

还有基于token的分割器：

````python
from langchain.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)

text1 = "foo bar bazzyfoo"
text_splitter.split_text(text1)
# 输出为：
# ['foo', ' bar', ' b', 'az', 'zy', 'foo']
````

还可以基于markdown文档的标题对其进行分割：

````python
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)

md_header_splits[0]
# 输出为：
# Document(page_content='Hi this is Jim  \nHi this is Joe', metadata={'Header 1': 'Title', 'Header 2': 'Chapter 1'})

md_header_splits[1]
# 输出为：
# Document(page_content='Hi this is Lance', metadata={'Header 1': 'Title', 'Header 2': 'Chapter 1', 'Header 3': 'Section'})
````

# 4.Vectorstores and Embedding

>强烈建议先看另一篇博客：[Question and Answer](https://shichaoxin.com/2025/06/06/LLM-LangChain-for-LLM-Application-Development/#5question-and-answer)作为基础。

````python
from langchain.document_loaders import PyPDFLoader

#载入数据
# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"), #注意：前两个PDF是一样的
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

#分割文档
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)
len(splits)
# 输出为：
# 209
````

先看一个将句子向量化的例子：

````python
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

np.dot(embedding1, embedding2) #输出为：0.9631511809630346
np.dot(embedding1, embedding3) #输出为：0.7702031371038216
np.dot(embedding2, embedding3) #输出为：0.7590540629791649
````

我们使用点积来比较两个向量的相似度，可以看到，第一个和第二个句子的相似度是最高的。接下来，我们将上述载入的已经拆分好的PDF文档进行向量化并存储到向量数据库中：

````python
#LangChain支持几十种不同的向量数据库
#Chroma是一个轻量级的向量数据库，可以直接在本地运行，无需服务器
from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'

#清空persist_directory
!rm -rf ./docs/chroma  # remove old database files if any

vectordb = Chroma.from_documents(
    documents=splits, #拆分后的文档
    embedding=embedding, #OpenAIEmbeddings模型
    persist_directory=persist_directory
)

print(vectordb._collection.count()) #与拆分的文档数量一样
# 输出为：
# 209
````

下面是一些使用示例：

````python
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question,k=3) #返回相似度前3的文档片段
len(docs) #输出为：3
docs[0].page_content #打印第一个文档片段的内容
# 输出为：
# "cs229-qa@cs.stanford.edu. This goes to an acc ount that's read by all the TAs and me. So \nrather than sending us email individually, if you send email to this account, it will \nactually let us get back to you maximally quickly with answers to your questions.  \nIf you're asking questions about homework probl ems, please say in the subject line which \nassignment and which question the email refers to, since that will also help us to route \nyour question to the appropriate TA or to me  appropriately and get the response back to \nyou quickly.  \nLet's see. Skipping ahead — let's see — for homework, one midterm, one open and term \nproject. Notice on the honor code. So one thi ng that I think will help you to succeed and \ndo well in this class and even help you to enjoy this cla ss more is if you form a study \ngroup.  \nSo start looking around where you' re sitting now or at the end of class today, mingle a \nlittle bit and get to know your classmates. I strongly encourage you to form study groups \nand sort of have a group of people to study with and have a group of your fellow students \nto talk over these concepts with. You can also  post on the class news group if you want to \nuse that to try to form a study group.  \nBut some of the problems sets in this cla ss are reasonably difficult.  People that have \ntaken the class before may tell you they were very difficult. And just I bet it would be \nmore fun for you, and you'd probably have a be tter learning experience if you form a"

#将向量数据库持久化保存到磁盘上，以便下次加载时无需重新计算或导入
vectordb.persist()
````

我们再看一个表现不好的例子：

````python
question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5) #返回相似度前5的文档片段
docs[0]
# 输出为：
# Document(page_content='those homeworks will be done in either MATLA B or in Octave, which is sort of — I \nknow some people call it a free ve rsion of MATLAB, which it sort  of is, sort of isn\'t.  \nSo I guess for those of you that haven\'t s een MATLAB before, and I know most of you \nhave, MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to \nplot data. And it\'s sort of an extremely easy to  learn tool to use for implementing a lot of \nlearning algorithms.  \nAnd in case some of you want to work on your  own home computer or something if you \ndon\'t have a MATLAB license, for the purposes of  this class, there\'s also — [inaudible] \nwrite that down [inaudible] MATLAB — there\' s also a software package called Octave \nthat you can download for free off the Internet. And it has somewhat fewer features than MATLAB, but it\'s free, and for the purposes of  this class, it will work for just about \neverything.  \nSo actually I, well, so yeah, just a side comment for those of you that haven\'t seen \nMATLAB before I guess, once a colleague of mine at a different university, not at \nStanford, actually teaches another machine l earning course. He\'s taught it for many years. \nSo one day, he was in his office, and an old student of his from, lik e, ten years ago came \ninto his office and he said, "Oh, professo r, professor, thank you so much for your', metadata={'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 8})
docs[1]
# 输出为：
# Document(page_content='those homeworks will be done in either MATLA B or in Octave, which is sort of — I \nknow some people call it a free ve rsion of MATLAB, which it sort  of is, sort of isn\'t.  \nSo I guess for those of you that haven\'t s een MATLAB before, and I know most of you \nhave, MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to \nplot data. And it\'s sort of an extremely easy to  learn tool to use for implementing a lot of \nlearning algorithms.  \nAnd in case some of you want to work on your  own home computer or something if you \ndon\'t have a MATLAB license, for the purposes of  this class, there\'s also — [inaudible] \nwrite that down [inaudible] MATLAB — there\' s also a software package called Octave \nthat you can download for free off the Internet. And it has somewhat fewer features than MATLAB, but it\'s free, and for the purposes of  this class, it will work for just about \neverything.  \nSo actually I, well, so yeah, just a side comment for those of you that haven\'t seen \nMATLAB before I guess, once a colleague of mine at a different university, not at \nStanford, actually teaches another machine l earning course. He\'s taught it for many years. \nSo one day, he was in his office, and an old student of his from, lik e, ten years ago came \ninto his office and he said, "Oh, professo r, professor, thank you so much for your', metadata={'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 8})
````

`docs[0]`和`docs[1]`是完全相同的两个文本片段（因为我们导入了两个一样的PDF），对于后续的LLM处理，引入了冗余重复的信息。接下来，我们再看另外一个表现不好的例子：

````python
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)
for doc in docs:
    print(doc.metadata)
# 输出为：
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 0}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 14}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture02.pdf', 'page': 0}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 6}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 8}

print(docs[4].page_content)
# 输出为：
# into his office and he said, "Oh, professo r, professor, thank you so much for your 
# machine learning class. I learned so much from it. There's this stuff that I learned in your 
# class, and I now use every day. And it's help ed me make lots of money, and here's a 
# picture of my big house."  
# So my friend was very excited. He said, "W ow. That's great. I'm glad to hear this 
# machine learning stuff was actually useful. So what was it that you learned? Was it 
# logistic regression? Was it the PCA? Was it the data ne tworks? What was it that you 
# learned that was so helpful?" And the student said, "Oh, it was the MATLAB."  
# So for those of you that don't know MATLAB yet, I hope you do learn it. It's not hard, 
# and we'll actually have a short MATLAB tutori al in one of the discussion sections for 
# those of you that don't know it.  
# Okay. The very last piece of logistical th ing is the discussion s ections. So discussion 
# sections will be taught by the TAs, and atte ndance at discussion sections is optional, 
# although they'll also be recorded and televi sed. And we'll use the discussion sections 
# mainly for two things. For the next two or th ree weeks, we'll use the discussion sections 
# to go over the prerequisites to this class or if some of you haven't seen probability or 
# statistics for a while or maybe algebra, we'll go over those in the discussion sections as a 
# refresher for those of you that want one.
````

我们的提问是想在第3节课中搜索一些关于回归的内容，但是输出的文档片段中有些并不属于第3节课，我们查看其输出的第1节课的文档片段，发现内容确实包含了回归。说明我们要求内容要出自第3节课这个需求并没有被完全捕获到。

# 5.Retrieval

本部分我们将采用一些新出现的高级方法来解决上一部分结尾提到的几个失败案例。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/5.png)

在第4部分中，我们是基于相似度在向量数据库中进行检索，其只考虑了文档片段的相似性。现在介绍另外一种检索方法：**MMR（Maximum Marginal Relevance）**，其会检索相似但尽量不冗余的文档片段，既考虑相似性又考虑多样性。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/6.png)

MMR的原理是先在向量数据库中按照相似度检索文档片段，得到`fetch_k`个最相似的文档片段，再从这`fetch_k`个文档片段中挑选出最具多样性的`k`个文档片段：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/7.png)

现在介绍第三种检索方法：自查询检索器（**self-query retriever**）：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/8.png)

在上述图中所示的例子中，输入句子首先会被查询解析器（查询解析器可以是一个LLM）分解为Filter和Search term两项，Filter用于筛选向量数据库中可以精确匹配的结果，Search term则会利用相似度检索进行模糊匹配。比如Filter会限定句子中出现的年份必须为1980年，而Search term则要求句子中必须有外星人相关的描述，但不一定非得包含Aliens这个词，也可以只包含比如ET这种外星人相关的词就行。

当我们从向量数据库中检索到很多相关的文档片段后，这些文档片段可能因为太长而无法塞进LLM的上下文窗口，因此我们需要对这些文档片段进行压缩。我们可以用另外一个LLM对这些文档片段进行压缩，比如删除无关部分的内容、只保留与问题相关的句子、直接生成摘要等。将压缩后的内容作为上下文送入LLM，再回答问题或生成文本。整个过程如下所示：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/9.png)

````python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
) #加载之前的向量数据库，见本文第4部分

print(vectordb._collection.count()) #输出为：209
````

我们先来看一个基于小的向量数据库的例子：

````python
#文本一共3句话，前两句基本是一样的信息
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

#构建一个小的向量数据库
smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Tell me about all-white mushrooms with large fruiting bodies"

#仅靠相似度检索
smalldb.similarity_search(question, k=2)
# 输出为：
# [Document(page_content='A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.', metadata={}),
#  Document(page_content='The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).', metadata={})]

#使用MMR检索方法
smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)
# 输出为：
# [Document(page_content='A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.', metadata={}),
#  Document(page_content='A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.', metadata={})]
````

回到之前的例子：

````python
question = "what did they say about matlab?"

#仅靠相似度检索，得到的前两个结果是完全一样的
docs_ss = vectordb.similarity_search(question,k=3)
docs_ss[0].page_content[:100]
# 输出为：
# 'those homeworks will be done in either MATLA B or in Octave, which is sort of — I \nknow some people '
docs_ss[1].page_content[:100]
# 输出为：
# 'those homeworks will be done in either MATLA B or in Octave, which is sort of — I \nknow some people '

#使用MMR检索方法，得到的前两个结果不一样
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
docs_mmr[0].page_content[:100]
# 输出为：
# 'those homeworks will be done in either MATLA B or in Octave, which is sort of — I \nknow some people '
docs_mmr[1].page_content[:100]
# 输出为：
# 'algorithm then? So what’s different? How come  I was making all that noise earlier about \nleast squa'
````

接下来是一个使用自查询检索器的例子：

````python
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)
#检索到的都是第3课的文本内容
for d in docs:
    print(d.metadata)
# 输出为：
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 0}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 14}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 4}
````

也可以调用已经封装好的自查询检索器接口：

````python
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

#对metadata中字段的结构说明
#我们导入的数据的metadata只包含source和page
#这些信息会被传递给LLM，所以description要尽可能详细
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

#文档内容的描述，帮助LLM更好地理解内容结构
document_content_description = "Lecture notes"

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
#构建自查询检索器
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "what did they say about regression in the third lecture?"

#自动解析query和filter
docs = retriever.get_relevant_documents(question)
# 打印信息：
# query='regression' filter=Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='source', value='docs/cs229_lectures/MachineLearning-Lecture03.pdf') limit=None

for d in docs:
    print(d.metadata)
# 输出为：
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 14}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 0}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 10}
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture03.pdf', 'page': 10}
````

接下来，我们可以对检索到的文档片段进行压缩：

````python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

# Wrap our vectorstore
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
compressor = LLMChainExtractor.from_llm(llm) #构建压缩器

#检索和压缩集成在一个接口里
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr") #默认按相似度进行检索，这里指定MMR
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)
# 输出为：
# Document 1:

# - "those homeworks will be done in either MATLA B or in Octave"
# - "I know some people call it a free ve rsion of MATLAB"
# - "MATLAB is I guess part of the programming language that makes it very easy to write codes using matrices, to write code for numerical routines, to move data around, to plot data."
# - "there's also a software package called Octave that you can download for free off the Internet."
# - "it has somewhat fewer features than MATLAB, but it's free, and for the purposes of this class, it will work for just about everything."
# - "once a colleague of mine at a different university, not at Stanford, actually teaches another machine learning course."
# ----------------------------------------------------------------------------------------------------
# Document 2:

# "Oh, it was the MATLAB."
# ----------------------------------------------------------------------------------------------------
# Document 3:

# - learning algorithms to teach a car how to drive at reasonably high speeds off roads avoiding obstacles.
# - that's a robot program med by PhD student Eva Roshen to teach a sort of somewhat strangely configured robot how to get on top of an obstacle, how to get over an obstacle.
# - So I think all of these are robots that I think are very difficult to hand-code a controller for by learning these sorts of learning algorithms.
# - Just a couple more last things, but let me just check what questions you have right now.
# - So if there are no questions, I'll just close with two reminders, which are after class today or as you start to talk with other people in this class, I just encourage you again to start to form project partners, to try to find project partners to do your project with.
# - And also, this is a good time to start forming study groups, so either talk to your friends or post in the newsgroup, but we just encourage you to try to start to do both of those today, okay? Form study groups, and try to find two other project partners.
````

目前为止，我们提到的这些检索技术都是基于向量数据库构建的。此外，还有一些其他的检索技术，完全不使用向量数据库，而是使用更加传统的NLP技术。比如：

````python
from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150) #RecursiveCharacterTextSplitter讲解见第3部分
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

question = "What are major topics for this class?"
docs_svm=svm_retriever.get_relevant_documents(question)
docs_svm[0]
# 输出为：
# Document(page_content="let me just check what questions you have righ t now. So if there are no questions, I'll just \nclose with two reminders, which are after class today or as you start to talk with other \npeople in this class, I just encourage you again to start to form project partners, to try to \nfind project partners to do your project with. And also, this is a good time to start forming \nstudy groups, so either talk to your friends  or post in the newsgroup, but we just \nencourage you to try to star t to do both of those today, okay? Form study groups, and try \nto find two other project partners.  \nSo thank you. I'm looking forward to teaching this class, and I'll see you in a couple of \ndays.   [End of Audio]  \nDuration: 69 minutes", metadata={})

question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
docs_tfidf[0]
# 输出为：
# Document(page_content="Saxena and Min Sun here did, wh ich is given an image like this, right? This is actually a \npicture taken of the Stanford campus. You can apply that sort of cl ustering algorithm and \ngroup the picture into regions. Let me actually blow that up so that you can see it more \nclearly. Okay. So in the middle, you see the lines sort of groupi ng the image together, \ngrouping the image into [inaudible] regions.  \nAnd what Ashutosh and Min did was they then  applied the learning algorithm to say can \nwe take this clustering and us e it to build a 3D model of the world? And so using the \nclustering, they then had a lear ning algorithm try to learn what the 3D structure of the \nworld looks like so that they could come up with a 3D model that you can sort of fly \nthrough, okay? Although many people used to th ink it's not possible to take a single \nimage and build a 3D model, but using a lear ning algorithm and that sort of clustering \nalgorithm is the first step. They were able to.  \nI'll just show you one more example. I like this  because it's a picture of Stanford with our \nbeautiful Stanford campus. So again, taking th e same sort of clustering algorithms, taking \nthe same sort of unsupervised learning algor ithm, you can group the pixels into different \nregions. And using that as a pre-processing step, they eventually built this sort of 3D model of Stanford campus in a single picture.  You can sort of walk  into the ceiling, look", metadata={})
````

# 6.Question Answering

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/10.png)

本部分是将第5部分检索到的文档片段喂给LLM生成最后的答案。建议先看另一篇博文：[【LLM】LangChain for LLM Application Development](https://shichaoxin.com/2025/06/06/LLM-LangChain-for-LLM-Application-Development/)。

````python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())
# 输出为：
# 209
question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)
# 输出为：
# 3

llm_name = "gpt-3.5-turbo"
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
result = qa_chain({"query": question})
result["result"]
# 输出为：
# 'The major topics for this class include machine learning, statistics, and algebra. Additionally, there will be discussions on extensions of the material covered in the main lectures.'

from langchain.prompts import PromptTemplate

# Build prompt
#{context}字段不能修改，LangChain内部会检索{context}和{question}这两个字段并完成内容替换
#{context}会被替换为retriever=vectordb.as_retriever()所检索出来的文档内容
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Is probability a class topic?"
result = qa_chain({"query": question})
result["result"]
# 输出为：
# 'Yes, probability is a class topic as the instructor assumes familiarity with basic probability and statistics. Thanks for asking!'
result["source_documents"][0]
# 输出为：
# Document(page_content="of this class will not be very program ming intensive, although we will do some \nprogramming, mostly in either MATLAB or Octa ve. I'll say a bit more about that later.  \nI also assume familiarity with basic proba bility and statistics. So most undergraduate \nstatistics class, like Stat 116 taught here at Stanford, will be more than enough. I'm gonna \nassume all of you know what ra ndom variables are, that all of you know what expectation \nis, what a variance or a random variable is. And in case of some of you, it's been a while \nsince you've seen some of this material. At some of the discussion sections, we'll actually \ngo over some of the prerequisites, sort of as  a refresher course under prerequisite class. \nI'll say a bit more about that later as well.  \nLastly, I also assume familiarity with basi c linear algebra. And again, most undergraduate \nlinear algebra courses are more than enough. So if you've taken courses like Math 51, \n103, Math 113 or CS205 at Stanford, that would be more than enough. Basically, I'm \ngonna assume that all of you know what matrix es and vectors are, that you know how to \nmultiply matrices and vectors and multiply matrix and matrices, that you know what a matrix inverse is. If you know what an eigenvect or of a matrix is, that'd be even better. \nBut if you don't quite know or if you're not qu ite sure, that's fine, too. We'll go over it in \nthe review sections.", metadata={'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 4})

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]
# 输出为：
# 'Yes, probability is a class topic in the document.'

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
result["result"]
# 输出为：
# 'The instructor, Andrew Ng, discusses the probabilistic interpretation of linear regression and how it can be used to derive the next learning algorithm, which will be the first classification algorithm. This algorithm will be used for classification problems where the value Y being predicted is a discrete value, such as binary classification where Y takes on only two values. Probability plays a crucial role in understanding and developing these classification algorithms. The discussion sections will also cover statistics and algebra as refreshers for those who need them, and will delve into extensions of the main lecture material to provide a more comprehensive understanding of machine learning concepts.'
````

如果我们想构建一个聊天机器人，则还需要模型可以记住之前的对话内容，但现在的QA chain还没有这个能力，比如下面这个例子，我们针对第一个问题的回答问了第二个问题，但模型给出的答案却与第一个问答毫不相关：

````python
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

question = "Is probability a class topic?"
result = qa_chain({"query": question})
result["result"]
# 输出为：
# 'Yes, probability is a class topic. The instructor assumes familiarity with basic probability and statistics for the course.'

question = "why are those prerequesites needed?"
result = qa_chain({"query": question})
result["result"]
# 输出为：
# 'The prerequisites for the class are needed because the course assumes that all students have a basic knowledge of computer science and computer skills. This foundational knowledge is essential for understanding the concepts and materials covered in the class, such as big-O notation and basic computer principles.'
````

# 7.Chat

>强烈建议先看另一篇博客[Memory](https://shichaoxin.com/2025/06/06/LLM-LangChain-for-LLM-Application-Development/#3memory)作为基础。

我们在第6部分介绍的`RetrievalQA`可以基于本地知识库进行问答，但却没有记忆能力，之前介绍的[`ConversationChain`](https://shichaoxin.com/2025/06/06/LLM-LangChain-for-LLM-Application-Development/#3memory)有记忆能力，可以进行多轮对话，但却不能基于本地知识库进行检索。因此，我们引入`ConversationalRetrievalChain`，结合了`RetrievalQA`和[`ConversationChain`](https://shichaoxin.com/2025/06/06/LLM-LangChain-for-LLM-Application-Development/#3memory)各自的特点，既能基于本地知识库，也可记忆多轮对话。

````python
llm_name = "gpt-3.5-turbo"

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever() #此处可以指定不同的检索方式
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "Is probability a class topic?"
result = qa({"question": question})
result['answer']
# 输出为：
# 'Yes, probability is a class topic. The instructor assumes familiarity with basic probability and statistics for the course.'

question = "why are those prerequesites needed?"
result = qa({"question": question})
result['answer'] #回答会基于第一个问答结果
# 输出为：
# 'Familiarity with basic probability and statistics is needed for the course because the material covered in the class involves concepts such as random variables, expectation, variance, and other statistical principles. Understanding these concepts is essential for grasping the machine learning algorithms and techniques that will be taught.'
````

最后，我们可以将上述所有流程集成到一个GUI中：

````python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

import panel as pn
import param

class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.loaded_file = "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
        self.qa = load_db(self.loaded_file,"stuff", 4)
    
    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style="outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query )
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self,count=0):
        self.chat_history = []
        return 

cb = cbfs()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput( placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp) 

jpg_pane = pn.pane.Image( './img/convchain.jpg')

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation,  loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2= pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources ),
)
tab3= pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4=pn.Column(
    pn.Row( file_input, button_load, bound_button_load),
    pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic" )),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
)
dashboard
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-chat-with-your-data/11.png)

感兴趣的可以自己运行代码尝试一下，还支持上传本地文档。

# 8.Conclusion

不再赘述。