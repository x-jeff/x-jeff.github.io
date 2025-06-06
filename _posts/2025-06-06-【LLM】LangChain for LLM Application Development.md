---
layout:     post
title:      【LLM】LangChain for LLM Application Development
subtitle:   LangChain应用开发入门
date:       2025-06-06
author:     x-jeff
header-img: blogimg/20200326.jpg
catalog: true
tags:
    - Large Language Models
---
>本文为参考吴恩达老师的"Building Systems with the ChatGPT API"课程所作的个人笔记。
>
>课程地址：[https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)。
>
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

LangChain：

* 是一个用于构建LLM应用程序的开源开发框架。
* 支持Python和JavaScript。
* 专注于组件和模块化。
* 核心价值：
    * 模块化组件。
    * 典型用例：常见的组件组合方式。

# 2.Models, Prompts and Parsers

````python
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = "gpt-3.5-turbo"

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]
````

首先，我们可以调用OpenAI的API将客户充满生气和愤怒的邮件转换为一种更为冷静和尊重的风格：

````python
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

response = get_completion(prompt)
````

模型输出的response为：

```
"Ah, I'm really frustrated that my blender lid flew off and splattered my kitchen walls with smoothie! And to make matters worse, the warranty doesn't cover the cost of cleaning up my kitchen. I could really use your help right now, friend."
```

接下来我们用LangChain封装好的API实现同样的功能：

````python
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(temperature=0.0, model=llm_model)
print(chat)
````

```
ChatOpenAI(verbose=False, callbacks=None, callback_manager=None, client=<class 'openai.api_resources.chat_completion.ChatCompletion'>, model_name='gpt-3.5-turbo', temperature=0.0, model_kwargs={}, openai_api_key=None, openai_api_base=None, openai_organization=None, request_timeout=None, max_retries=6, streaming=False, n=1, max_tokens=None)
```

我们可以通过ChatOpenAI设置很多模型相关的参数。然后我们可以使用LangChain提供的提示词模板功能来批量处理类似的请求：

````python
#这里不需要f-string了
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)
print(prompt_template.messages[0].prompt)
````

````
PromptTemplate(input_variables=['style', 'text'], output_parser=None, partial_variables={}, template='Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```\n', template_format='f-string', validate_template=True)
````

````python
customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

print(type(customer_messages))
print(type(customer_messages[0]))
print(customer_messages[0])
````

````
<class 'list'>
<class 'langchain.schema.HumanMessage'>
content="Translate the text that is delimited by triple backticks into a style that is American English in a calm and respectful tone\n. text: ```\nArrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!\n```\n" additional_kwargs={} example=False
````

````python
# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)
print(customer_response.content)
````

```
Oh man, I'm really frustrated that my blender lid flew off and made a mess of my kitchen walls with smoothie! And on top of that, the warranty doesn't cover the cost of cleaning up my kitchen. I could really use your help right now, buddy!
```

通过提示词模板功能，我们可以很方便的将客服的回复转化为我们想要的风格：

````python
service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

service_response = chat(service_messages)
print(service_response.content)
````

```
Ahoy there, valued customer! Regrettably, the warranty be not coverin' the cost o' cleanin' yer galley due to yer own negligence. Ye see, 'twas yer own doin' when ye forgot to secure the lid afore startin' the blender. 'Tis a tough break, indeed! Fare thee well, matey!
```

LangChain还可以对输出进行解析。比如我们希望模型输出以下的JSON格式：

```json
{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}
```

我们先来看下不用输出解析的情况，下面是一个用户的商品评价以及我们提供的提示词：

````python
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""
````

我们还用之前刚学的那套流程得到模型输出：

````python
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)
messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=llm_model)
response = chat(messages)
print(response.content)
````

````
{
  "gift": true,
  "delivery_days": 2,
  "price_value": "It's slightly more expensive than the other leaf blowers out there"
}
````

如果我们检查下`response`的类型，可以发现它是一个`str`，并不是一个Python字典：

````python
type(response.content) #输出为：str
````

如果我们想通过key-value的方式取值时，程序会报错：

````python
# You will get an error by running this line of code 
# because'gift' is not a dictionary
# 'gift' is a string
response.content.get('gift')
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/1.png)

我们可以借助LangChain的结果解析功能来解决这个问题：

````python
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
print(format_instructions)
````

````
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "\`\`\`json" and "\`\`\`":

```json
{
	"gift": string  // Was the item purchased                             as a gift for someone else?                              Answer True if yes,                             False if not or unknown.
	"delivery_days": string  // How many days                                      did it take for the product                                      to arrive? If this                                       information is not found,                                      output -1.
	"price_value": string  // Extract any                                    sentences about the value or                                     price, and output them as a                                     comma separated Python list.
}
```
````

`format_instructions`的使用方法见下：

````python
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions) #此处可以指定format_instructions

response = chat(messages)
print(response.content)
````

````
```json
{
	"gift": true,
	"delivery_days": 2,
	"price_value": ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]
}
```
````

将结果解析为Python字典格式：

````python
output_dict = output_parser.parse(response.content)
print(output_dict)
print(type(output_dict))
````

````
{'gift': True,
 'delivery_days': 2,
 'price_value': ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]}
dict
````

这时候我们再通过key-value的形式取值就没问题了：

````python
output_dict.get('delivery_days') #输出为：2
````

# 3.Memory

本部分着重介绍如何利用LangChain让模型记住之前的对话内容。

````python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory, #记住之前的多轮对话
    verbose=False #True会输出更多细节信息，False就只输出最后结果
)
````

然后我们进行多轮对话测试一下：

````python
conversation.predict(input="Hi, my name is Andrew")
````

````
"Hello Andrew! It's nice to meet you. How can I assist you today?"
````

````python
conversation.predict(input="What is 1+1?")
````

````
'1+1 equals 2. Is there anything else you would like to know?'
````

````python
conversation.predict(input="What is my name?")
````

````
'Your name is Andrew.'
````

看来模型成功记住了之前的对话内容。我们把`verbose`设为True，重新执行一遍这个对话，看下更多的细节信息：

````python
conversation.predict(input="Hi, my name is Andrew")
````

````
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi, my name is Andrew
AI:

> Finished chain.
"Hello Andrew! It's nice to meet you. How can I assist you today?"
````

````python
conversation.predict(input="What is 1+1?")
````

````
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is Andrew
AI: Hello Andrew! It's nice to meet you. How can I assist you today?
Human: What is 1+1?
AI:

> Finished chain.
'1+1 equals 2. Is there anything else you would like to know?'
````

````python
conversation.predict(input="What is my name?")
````

````
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is Andrew
AI: Hello Andrew! It's nice to meet you. How can I assist you today?
Human: What is 1+1?
AI: 1+1 equals 2. Is there anything else you would like to know?
Human: What is my name?
AI:

> Finished chain.
'Your name is Andrew.'
````

可以查看之前的对话内容：

````python
print(memory.buffer)
````

````
Human: Hi, my name is Andrew
AI: Hello Andrew! It's nice to meet you. How can I assist you today?
Human: What is 1+1?
AI: 1+1 equals 2. Is there anything else you would like to know?
Human: What is my name?
AI: Your name is Andrew.
````

````python
memory.load_memory_variables({})
````

````
{'history': "Human: Hi, my name is Andrew\nAI: Hello Andrew! It's nice to meet you. How can I assist you today?\nHuman: What is 1+1?\nAI: 1+1 equals 2. Is there anything else you would like to know?\nHuman: What is my name?\nAI: Your name is Andrew."}
````

可见，之前所有的历史对话都保存在memory中，我们可以新建一个memory，并手动添加历史对话：

````python
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
print(memory.buffer)
````

````
Human: Hi
AI: What's up
````

````python
memory.load_memory_variables({})
````

````
{'history': "Human: Hi\nAI: What's up"}
````

````python
memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})
memory.load_memory_variables({})
````

````
{'history': "Human: Hi\nAI: What's up\nHuman: Not much, just hanging\nAI: Cool"}
````

随着对话越来越长，memory中保存的上下文内容也越来越多，每次都会发送更多的token给LLM处理，这会造成费用越来越高。因此，我们可以通过LangChain提供的`ConversationBufferWindowMemory`来限制memory中保存的对话轮次，比如我们设`k=1`，表示memory只保存最新一轮的对话内容：

````python
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1)

memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

memory.load_memory_variables({})
````

````
{'history': 'Human: Not much, just hanging\nAI: Cool'}
````

可以看到，memory中只保存了最近一次的对话。如果我们重复之前的对话，就会得到如下的结果：

````python
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1) #只保存最近一轮对话
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)
````

````python
conversation.predict(input="Hi, my name is Andrew")
````

````
"Hello Andrew! It's nice to meet you. How can I assist you today?"
````

````python
conversation.predict(input="What is 1+1?")
````

````
'1+1 equals 2. Is there anything else you would like to know?'
````

````python
conversation.predict(input="What is my name?")
````

````
"I'm sorry, I do not have access to personal information such as your name. Is there anything else you would like to know?"
````

除了限制保存的对话轮数，也可以直接限制memory中保存的token数量：

````python
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0, model=llm_model)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

memory.load_memory_variables({})
````

````
{'history': 'AI: Amazing!\nHuman: Backpropagation is what?\nAI: Beautiful!\nHuman: Chatbots are what?\nAI: Charming!'}
````

可以看到，memory中并没有保存所有的历史对话，只是保存了最近的50个token。

到目前，我们介绍了三种memory方式，第一种是保存所有的历史对话（`ConversationBufferMemory`），第二种是限制保存的历史对话轮数（`ConversationBufferWindowMemory`），第三种是限制保存的token数量（`ConversationTokenBufferMemory`）。现在介绍第四种方式，即让LLM对历史对话进行总结，然后将总结保存在memory中（`ConversationSummaryBufferMemory`），我们来看一个例子：

````python
from langchain.memory import ConversationSummaryBufferMemory

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

memory.load_memory_variables({})
````

````
{'history': 'System: The human and AI exchange greetings and discuss the schedule for the day, including a meeting with the product team, work on the LangChain project, and a lunch meeting with a customer interested in AI. The AI provides details on each event and emphasizes the power of LangChain as a tool.'}
````

可以看到，因为历史对话的所有内容超出了100个token的数量限制，所以历史对话被LLM总结，并以`System`的角色保存在memory中。下面来测试一下：

````python
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="What would be a good demo to show?")
````

````
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
System: The human and AI exchange greetings and discuss the schedule for the day, including a meeting with the product team, work on the LangChain project, and a lunch meeting with a customer interested in AI. The AI provides details on each event and emphasizes the power of LangChain as a tool.
Human: What would be a good demo to show?
AI:

> Finished chain.
'For the meeting with the product team, a demo showcasing the latest features and updates on the LangChain project would be ideal. This could include a live demonstration of how LangChain streamlines language translation processes, improves accuracy, and increases efficiency. Additionally, highlighting any recent success stories or case studies would be beneficial to showcase the real-world impact of LangChain.'
````

除此之外，LangChain还支持其他很多种memory方式，在此不再一一详述。

# 4.Chains

本部分将介绍LangChain中最关键的概念：chain。chain通常将LLM和提示词结合在一起构成一个building block，然后我们可以将多个building block组合在一起，对文本或其他数据执行一系列操作。

首先，我们加载下待会要用到的数据：

````python
import pandas as pd
df = pd.read_csv('Data.csv')

print(df.head())
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/2.png)

第一列是产品名称，第二列是对产品的评价。

## 4.1.LLMChain

LLMChain是一个简单但非常强大的chain，也支撑着我们将来要讨论的许多chain。

````python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9, model=llm_model)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
````

将LLM和提示词组合成一个LLMChain：

````python
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)
````

````
'"Royal Rest Bedding Co."'
````

## 4.2.Sequential Chains

sequential chain会组合多个chain，其特点是一个chain的输出是下一个chain的输入。sequential chain分为两种类型：

* SimpleSequentialChain：单个输入/输出。
* SequentialChain：多个输入/输出。

### 4.2.1.SimpleSequentialChain

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/5.png)

每个子chain只有单个输入和单个输出。

````python
from langchain.chains import SimpleSequentialChain

llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt) #第一个子chain

# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt) #第二个子chain

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )

overall_simple_chain.run(product)
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/3.png)

子chain按顺序逐个运行。

### 4.2.2.SequentialChain

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/6.png)

当处理多个输入或多个输出时，可以使用常规的SequentialChain。

````python
from langchain.chains import SequentialChain

llm = ChatOpenAI(temperature=0.9, model=llm_model)

#定义第一个chain
# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )

#定义第二个chain
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    ) #chain2的输入来自chain1的输出

#定义第三个chain
# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )

#定义第四个chain
# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     ) #chain4的输入来自chain2和chain3的输出

#定义SequentialChain
# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)

review = df.Review[5]
overall_chain(review)
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/4.png)

## 4.3.Router Chain

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/7.png)

对于更复杂的操作，我们可以使用router chain，其可以根据输入，执行不同的子chain。

举个例子，我们定义4个提示词模板：第一个模板擅长回答物理相关的问题；第二个模板擅长回答数学相关的问题；第三个模板擅长回答历史相关的问题；第四个模板擅长回答计算机科学相关的问题：

````python
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""
````

提供更多的信息：

````python
prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]
````

这些信息将会被传递给router chain。导入一些必要的库：

````python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
````

定义`destination_chains`：

````python
llm = ChatOpenAI(temperature=0, model=llm_model)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
````

定义`default_chain`：

````python
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
````

定义`router_chain`：

````python
#用于router chain的提示词模板
#输入：用户提供
#输出：JSON格式，有两个key：destination和next_inputs
#第一个key：destination的含义：
#让模型根据用户输入自行选择destination_chains中合适的子chain执行
#如果没有合适的子chain，则执行default_chain
#destination可以是physics、math、History、computer science、DEFAULT
#第二个key：next_inputs的含义：
#如果模型认为修改用户输入可以得到更好的结果，则模型可以修改输入，并输出在next_inputs中
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{
    "destination": string \ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \ a potentially modified version of the original input
}
```

REMEMBER: The value of “destination” MUST match one of \
the candidate prompts listed below.\
If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

#定义router chain的提示词模板
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt) #定义router chain
````

`RouterOutputParser()`会将模型输出解析成一个python字典，字典中至少包含两个key：

1. `destination`：用于指定下一个要执行的子chain。
2. `next_inputs`：下一个子chain的输入。

创建完整的chain：

````python
chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )
````

参数解释：

1. `router_chain`：接收用户输入并输出目标名称（必须与`destination_chains`的key匹配），决定后续使用哪个子chain。
2. `destination_chains`：key-value的形式，key为目标名称，value为对应的子chain。
3. `default_chain`：当`router_chain`输出的目标名称不在`destination_chains`时，执行该默认子chain。

实际测试一下：

````python
chain.run("What is black body radiation?")
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/8.png)

````python
chain.run("what is 2 + 2")
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/9.png)

````python
chain.run("Why does every cell in our body contain DNA?")
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/10.png)

# 5.Question and Answer

本部分我们将基于PDF文件、网页或者公司内部文档等不在模型训练集内的一些文本，构建一个LLM问答系统。

````python
from langchain.chains import RetrievalQA #用于在文档上进行一些检索
from langchain.chat_models import ChatOpenAI #OpenAI模型
from langchain.document_loaders import CSVLoader #加载csv文件
from langchain.vectorstores import DocArrayInMemorySearch #向量存储
from IPython.display import display, Markdown #用于显示
from langchain.llms import OpenAI
````

导入我们要用的CSV文件：

````python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
````

CSV文件中的内容如下所示（仅展示前5行），第一列是索引，第二列是商品名称，第三列是商品描述。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/11.png)

我们可以通过调用`VectorstoreIndexCreator`来一步到位的构建基于本地文档的LLM问答系统：

````python
from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)

display(Markdown(response))
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/12.png)

`OpenAI`适用于传统文本生成任务，即单次提示。而之前使用的`ChatOpenAI`更适用于多轮对话场景。两者调用的OpenAI API也不同。

`VectorstoreIndexCreator`封装了以下一些核心步骤：

1. 文档加载。
2. 文本拆分。因为LLM通常无法一次性处理过长的文本，所以通常需要对文本进行拆分。
3. 向量化嵌入，指的是将拆分后的文本转化为向量。注意和[词嵌入](https://shichaoxin.com/2021/01/17/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E5%9B%9B%E5%8D%81%E4%BA%94%E8%AF%BE-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B8%8E%E8%AF%8D%E5%B5%8C%E5%85%A5/)这个概念进行区分，[词嵌入](https://shichaoxin.com/2021/01/17/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80-%E7%AC%AC%E5%9B%9B%E5%8D%81%E4%BA%94%E8%AF%BE-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B8%8E%E8%AF%8D%E5%B5%8C%E5%85%A5/)通常是将token转化为向量，而向量化嵌入是一个更宽泛的概念，它可以对任意文本单元（可以是词、子词、句子、段落、整篇文档）甚至图像、音频等多模态数据进行向量映射。
4. 向量存储。不单单是把向量保存起来，还要考虑后续的相似度检索、高效更新、大规模部署等问题。
5. 索引与检索，即找出与给定查询最相近的文档片段。

针对第2步，LLM一次通常只能处理几千个词，所以需要对长文本进行拆分（chunks）：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/13.png)

针对第3步，对拆分后的文本进行向量化，相似的文本将得到相似的向量：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/14.png)

比如下面3个句子，前两个句子都是关于宠物的，而第三个句子则是关于汽车的，所以前两个句子的相似度会更高：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/15.png)

针对第4步，我们将这些向量保存在向量数据库中：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/16.png)

针对第5步，当有查询进来时，我们首先会为查询创建一个嵌入向量，然后将其与向量数据库中的所有向量进行比较，并选择前n个最相似的，最后将它们返回给语言模型以获得最终答案：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/17.png)

接下来我们用代码分步实现这些核心步骤。

首先是第一步，文档加载：

````python
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)

docs = loader.load()

print(docs[0])
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/18.png)

````python
print(docs[1])
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/19.png)

`docs`中的每个元素对应CSV文件中的一行。因为我们输入的CSV文档并不长，所以这里就不需要对文本进行拆分了，我们直接跳到第三步。先来看一个向量化的例子：

````python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("Hi my name is Harrison")

print(len(embed))
print(embed[:5])
````

````
1536
[-0.021964654326438904, 0.006758837960660458, -0.01824948936700821, -0.03923514857888222, -0.014007173478603363]
````

这里我们使用了OpenAI提供的Embedding API。句子"Hi my name is Harrison"被转换成了一个长度为1536的向量。接下来，我们对`docs`进行向量化并进行存储：

````python
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
````

现在，我们可以传入一个查询，并在向量数据库中找到最相似的几个文本片段：

````python
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
print(len(docs))
print(docs[0])
````

````
4
Document(page_content=': 255\nname: Sun Shield Shirt by\ndescription: "Block the sun, not the fun – our high-performance sun shirt is guaranteed to protect from harmful UV rays. \n\nSize & Fit: Slightly Fitted: Softly shapes the body. Falls at hip.\n\nFabric & Care: 78% nylon, 22% Lycra Xtra Life fiber. UPF 50+ rated – the highest rated sun protection possible. Handwash, line dry.\n\nAdditional Features: Wicks moisture for quick-drying comfort. Fits comfortably over your favorite swimsuit. Abrasion resistant for season after season of wear. Imported.\n\nSun Protection That Won\'t Wear Off\nOur high-performance fabric provides SPF 50+ sun protection, blocking 98% of the sun\'s harmful rays. This fabric is recommended by The Skin Cancer Foundation as an effective UV protectant.', metadata={'source': 'OutdoorClothingCatalog_1000.csv', 'row': 255})
````

可以看到，一共找到了4个最相似的文本片段。我们将这4个文本片段连接在一起后交给LLM去总结：

````python
#尝试使用gpt-3.5-turbo发现并不能输出markdown table格式，所以换用了gpt-4
llm = ChatOpenAI(temperature = 0.0, model = "gpt-4")
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")
display(Markdown(response))
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/20.png)

我们可以将上面两步（即相似度查询和LLM处理）封装起来，首先，将向量数据库`db`包装成符合LangChain的Retriever接口对象，方便后续统一调用：

````python
retriever = db.as_retriever()
````

然后调用`RetrievalQA`创建一个QA chain，一键实现对查询文本的相似度检索和LLM处理：

````python
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
````

参数`chain_type`可指定多种“文档汇总+问答”策略，其中包括：

1. `"stuff"`：会把所有检索到的文档原封不动的全塞到一个大的prompt里，然后一起送给LLM生成答案，这也是我们之前分步实现时的策略。

    ![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/22.png)

2. `"map_reduce"`：map指的是对每个检索到的文档片段分别让LLM生成一个“局部答案”，reduce是指再把所有“局部答案”聚合、总结成一个全局答案。

    ![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/23.png)

3. `"refine"`：用前N份文档生成一个初始答案，对于后续每一份文档，将其与已有答案一起送入LLM，让模型去优化答案，反复迭代，直到所有文档都被处理完。

    ![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/24.png)

4. `"map_rerank"`：对每个文档片段分别生成候选答案，用模型对这些候选答案按相关性或质量打分排序，选取最优的若干候选，再聚合成最终回答。

    ![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/25.png)

我们使用和之前一样的查询测试一下：

````python
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
display(Markdown(response))
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/21.png)

通过这些分步，我们可以得到和`VectorstoreIndexCreator`一样的结果。

# 6.Evaluation

本部分主要介绍如何评估LLM应用的好坏。我们以第5部分构建的LLM应用为例：

````python
#这段代码和第5部分一样
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature = 0.0, model="gpt-4")
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
````

我们可以人工构建一些问答对来评估LLM的表现，比如：

````python
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
````

但是人工构建这些问答对过于繁琐和耗时，我们可以借助LLM帮我们自动生成一些问答对：

````python
from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
print(new_examples)
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/26.png)

LLM一共帮我们产生了5个问答对。我们可以把之前人工构建的2个问答对和这5个自动生成的问答对放在一起，相当于是ground truth，然后让我们的LLM应用基于每个问答对中的问题生成自己的答案，随后将这些答案与问答对中的标准答案进行比对，从而实现对LLM应用的评估：

````python
examples += new_examples #一共7个问答对
qa.run(examples[0]["query"]) #让LLM应用对第一个问题产生自己的答案
````

````
> Entering new RetrievalQA chain...

> Finished chain.
'Yes, the Cozy Comfort Pullover Set does have side pockets.'
````

额外补充下，我们可以开启debug模式，这样就能看到整个chain是怎么运行的了：

````python
import langchain
langchain.debug = True #开启debug模式

qa.run(examples[0]["query"]) #依然以第一个问题为例
````

````
[chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
{
  "query": "Do the Cozy Comfort Pullover Set        have side pockets?"
}
[chain/start] [1:chain:RetrievalQA > 2:chain:StuffDocumentsChain] Entering Chain run with input:
[inputs]
[chain/start] [1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain] Entering Chain run with input:
{
  "question": "Do the Cozy Comfort Pullover Set        have side pockets?",
  "context": ": 10\nname: Cozy Comfort Pullover Set, Stripe\ndescription: Perfect for lounging, this striped knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out.\n\nSize & Fit\n- Pants are Favorite Fit: Sits lower on the waist.\n- Relaxed Fit: Our most generous fit sits farthest from the body.\n\nFabric & Care\n- In the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features\n- Relaxed fit top with raglan sleeves and rounded hem.\n- Pull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg.\n\nImported.<<<<>>>>>: 73\nname: Cozy Cuddles Knit Pullover Set\ndescription: Perfect for lounging, this knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out. \n\nSize & Fit \nPants are Favorite Fit: Sits lower on the waist. \nRelaxed Fit: Our most generous fit sits farthest from the body. \n\nFabric & Care \nIn the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features \nRelaxed fit top with raglan sleeves and rounded hem. \nPull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg. \nImported.<<<<>>>>>: 632\nname: Cozy Comfort Fleece Pullover\ndescription: The ultimate sweater fleece \u2013 made from superior fabric and offered at an unbeatable price. \n\nSize & Fit\nSlightly Fitted: Softly shapes the body. Falls at hip. \n\nWhy We Love It\nOur customers (and employees) love the rugged construction and heritage-inspired styling of our popular Sweater Fleece Pullover and wear it for absolutely everything. From high-intensity activities to everyday tasks, you'll find yourself reaching for it every time.\n\nFabric & Care\nRugged sweater-knit exterior and soft brushed interior for exceptional warmth and comfort. Made from soft, 100% polyester. Machine wash and dry.\n\nAdditional Features\nFeatures our classic Mount Katahdin logo. Snap placket. Front princess seams create a feminine shape. Kangaroo handwarmer pockets. Cuffs and hem reinforced with jersey binding. Imported.\n\n \u2013 Official Supplier to the U.S. Ski Team\nTHEIR WILL TO WIN, WOVEN RIGHT IN. LEARN MORE<<<<>>>>>: 151\nname: Cozy Quilted Sweatshirt\ndescription: Our sweatshirt is an instant classic with its great quilted texture and versatile weight that easily transitions between seasons. With a traditional fit that is relaxed through the chest, sleeve, and waist, this pullover is lightweight enough to be worn most months of the year. The cotton blend fabric is super soft and comfortable, making it the perfect casual layer. To make dressing easy, this sweatshirt also features a snap placket and a heritage-inspired Mt. Katahdin logo patch. For care, machine wash and dry. Imported."
}
[llm/start] [1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain > 4:llm:ChatOpenAI] Entering LLM run with input:
{
  "prompts": [
    "System: Use the following pieces of context to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n: 10\nname: Cozy Comfort Pullover Set, Stripe\ndescription: Perfect for lounging, this striped knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out.\n\nSize & Fit\n- Pants are Favorite Fit: Sits lower on the waist.\n- Relaxed Fit: Our most generous fit sits farthest from the body.\n\nFabric & Care\n- In the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features\n- Relaxed fit top with raglan sleeves and rounded hem.\n- Pull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg.\n\nImported.<<<<>>>>>: 73\nname: Cozy Cuddles Knit Pullover Set\ndescription: Perfect for lounging, this knit set lives up to its name. We used ultrasoft fabric and an easy design that's as comfortable at bedtime as it is when we have to make a quick run out. \n\nSize & Fit \nPants are Favorite Fit: Sits lower on the waist. \nRelaxed Fit: Our most generous fit sits farthest from the body. \n\nFabric & Care \nIn the softest blend of 63% polyester, 35% rayon and 2% spandex.\n\nAdditional Features \nRelaxed fit top with raglan sleeves and rounded hem. \nPull-on pants have a wide elastic waistband and drawstring, side pockets and a modern slim leg. \nImported.<<<<>>>>>: 632\nname: Cozy Comfort Fleece Pullover\ndescription: The ultimate sweater fleece \u2013 made from superior fabric and offered at an unbeatable price. \n\nSize & Fit\nSlightly Fitted: Softly shapes the body. Falls at hip. \n\nWhy We Love It\nOur customers (and employees) love the rugged construction and heritage-inspired styling of our popular Sweater Fleece Pullover and wear it for absolutely everything. From high-intensity activities to everyday tasks, you'll find yourself reaching for it every time.\n\nFabric & Care\nRugged sweater-knit exterior and soft brushed interior for exceptional warmth and comfort. Made from soft, 100% polyester. Machine wash and dry.\n\nAdditional Features\nFeatures our classic Mount Katahdin logo. Snap placket. Front princess seams create a feminine shape. Kangaroo handwarmer pockets. Cuffs and hem reinforced with jersey binding. Imported.\n\n \u2013 Official Supplier to the U.S. Ski Team\nTHEIR WILL TO WIN, WOVEN RIGHT IN. LEARN MORE<<<<>>>>>: 151\nname: Cozy Quilted Sweatshirt\ndescription: Our sweatshirt is an instant classic with its great quilted texture and versatile weight that easily transitions between seasons. With a traditional fit that is relaxed through the chest, sleeve, and waist, this pullover is lightweight enough to be worn most months of the year. The cotton blend fabric is super soft and comfortable, making it the perfect casual layer. To make dressing easy, this sweatshirt also features a snap placket and a heritage-inspired Mt. Katahdin logo patch. For care, machine wash and dry. Imported.\nHuman: Do the Cozy Comfort Pullover Set        have side pockets?"
  ]
}
[llm/end] [1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain > 4:llm:ChatOpenAI] [17.971ms] Exiting LLM run with output:
{
  "generations": [
    [
      {
        "text": "Yes, the Cozy Comfort Pullover Set does have side pockets.",
        "generation_info": null,
        "message": {
          "content": "Yes, the Cozy Comfort Pullover Set does have side pockets.",
          "additional_kwargs": {},
          "example": false
        }
      }
    ]
  ],
  "llm_output": {
    "token_usage": {
      "prompt_tokens": 732,
      "completion_tokens": 14,
      "total_tokens": 746
    },
    "model_name": "gpt-4"
  }
}
[chain/end] [1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain] [18.450999999999997ms] Exiting Chain run with output:
{
  "text": "Yes, the Cozy Comfort Pullover Set does have side pockets."
}
[chain/end] [1:chain:RetrievalQA > 2:chain:StuffDocumentsChain] [18.881999999999998ms] Exiting Chain run with output:
{
  "output_text": "Yes, the Cozy Comfort Pullover Set does have side pockets."
}
[chain/end] [1:chain:RetrievalQA] [65.87700000000001ms] Exiting Chain run with output:
{
  "result": "Yes, the Cozy Comfort Pullover Set does have side pockets."
}
'Yes, the Cozy Comfort Pullover Set does have side pockets.'
````

可以看到，根据问题，一共在文档中搜索到了4个最相关的文档片段。此外，debug模式还能输出所用的token数量，便于成本控制。关闭debug模式只需要：

````python
# Turn off the debug mode
langchain.debug = False
````

言归正传，现在我们让LLM应用对所有的7个问题都生成自己的答案：

````python
predictions = qa.apply(examples)
````

````
> Entering new RetrievalQA chain...

> Finished chain.


> Entering new RetrievalQA chain...

> Finished chain.


> Entering new RetrievalQA chain...

> Finished chain.


> Entering new RetrievalQA chain...

> Finished chain.


> Entering new RetrievalQA chain...

> Finished chain.


> Entering new RetrievalQA chain...

> Finished chain.


> Entering new RetrievalQA chain...

> Finished chain.
````

从上述打印的信息可以看到，7个问题被逐个处理。现在我们有了ground truth和预测结果，需要对两者进行比对从而完成对LLM应用的评估，这一过程我们依然可以借助LLM完成，让LLM去评估ground truth和预测结果是否一致：

````python
from langchain.evaluation.qa import QAEvalChain
llm = ChatOpenAI(temperature=0, model=llm_model) #用于评估结果的LLM
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
````

````
Example 0:
Question: Do the Cozy Comfort Pullover Set        have side pockets?
Real Answer: Yes
Predicted Answer: Yes, the Cozy Comfort Pullover Set does have side pockets.
Predicted Grade: CORRECT

Example 1:
Question: What collection is the Ultra-Lofty         850 Stretch Down Hooded Jacket from?
Real Answer: The DownTek collection
Predicted Answer: The Ultra-Lofty 850 Stretch Down Hooded Jacket is from the DownTek collection.
Predicted Grade: CORRECT

Example 2:
Question: What is the weight of each pair of Women's Campside Oxfords?
Real Answer: The approximate weight of each pair of Women's Campside Oxfords is 1 lb. 1 oz.
Predicted Answer: The approximate weight of each pair of Women's Campside Oxfords is 1 lb. 1 oz.
Predicted Grade: CORRECT

Example 3:
Question:  What are the dimensions of the small and medium sizes for the Recycled Waterhog Dog Mat, Chevron Weave?


Real Answer:  The small size has dimensions of 18" x 28" and the medium size has dimensions of 22.5" x 34.5".
Predicted Answer: The small Recycled Waterhog Dog Mat, Chevron Weave has dimensions of 18" x 28". The medium size has dimensions of 22.5" x 34.5".
Predicted Grade: CORRECT

Example 4:
Question:  What are some key features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece as described in the document?


Real Answer:  The key features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece include bright colors, ruffles, exclusive whimsical prints, four-way-stretch and chlorine-resistant fabric, UPF 50+ rated fabric for sun protection, crossover no-slip straps, fully lined bottom for secure fit and maximum coverage.
Predicted Answer: The Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece has several key features. It has bright colors, ruffles, and exclusive whimsical prints. The fabric is four-way-stretch and chlorine-resistant, which helps it keep its shape and resist snags. The fabric is also UPF 50+ rated, providing the highest rated sun protection possible and blocking 98% of the sun's harmful rays. The swimsuit has crossover no-slip straps and a fully lined bottom for a secure fit and maximum coverage. It is recommended to machine wash and line dry the swimsuit for best results.
Predicted Grade: CORRECT

Example 5:
Question:  What is the fabric composition of the Refresh Swimwear V-Neck Tankini Contrasts?


Real Answer:  The body of the tankini top is made of 82% recycled nylon and 18% Lycra® spandex, while the lining is made of 90% recycled nylon and 10% Lycra® spandex.
Predicted Answer: The Refresh Swimwear, V-Neck Tankini Contrasts is made of 82% recycled nylon with 18% Lycra® spandex. It is lined with 90% recycled nylon and 10% Lycra® spandex.
Predicted Grade: CORRECT

Example 6:
Question:  What technology sets the EcoFlex 3L Storm Pants apart from other waterproof pants?


Real Answer:  The EcoFlex 3L Storm Pants feature TEK O2 technology, which offers the most breathability ever tested in waterproof pants.
Predicted Answer: The EcoFlex 3L Storm Pants are set apart by the TEK O2 technology. This state-of-the-art technology offers the most breathability ever tested, making the pants suitable for a variety of outdoor activities year-round. The pants are also loaded with features outdoor enthusiasts appreciate, including weather-blocking gaiters and handy side zips.
Predicted Grade: CORRECT
````

# 7.Agents

Agent是LangChain中最核心的高级组件之一。Agent是一个能够根据用户输入动态决定调用哪些工具来完成任务的智能体，而不是固定地执行一个预定义的chain。所调用的这些工具的本质就是一个python函数包装的API调用接口。这些python函数可以使用LangChain预定义好的，也可以自定义。比如LangChain预定义好的`"wikipedia"`工具，其会调用维基百科的python API进行查询。简单总结下Agent的工作原理：

1. 应用LLM解析用户输入。
2. 决定调用哪个工具（从已载入的工具中选择）。
3. 执行工具。
4. LLM处理工具返回的结果。
5. 输出最终结果给用户。

接下来看几个例子。

````python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
tools = load_tools(["llm-math","wikipedia"], llm=llm) #载入两个工具
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True) #初始化一个agent
````

`initialize_agent`用于初始化一个Agent。

参数`agent`用于指定Agent的类型，最常用的一种类型就是`CHAT_ZERO_SHOT_REACT_DESCRIPTION`，其中，`CHAT`的含义是使用chat类模型（比如GPT-3.5，GPT4等，支持多轮对话格式）；`ZERO_SHOT`的含义是不需要提供示例，模型可以根据工具的描述来自主决定使用哪个工具；`REACT`表示采用ReAct框架：Reasoning+Action；`DESCRIPTION`表示每个工具的描述会成为LLM判断是否调用的重要依据，比如工具`"llm-math"`的描述是"Useful for when you need to answer questions about math."，工具`"wikipedia"`的描述是"A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query."。

参数`handle_parsing_errors`用于判断当LLM输出内容格式不符合预期时，是否让Agent自动忽略并继续执行。设为True可以让Agent更健壮，不至于因为格式错误中断整个推理流程。

````python
agent("What is the 25% of 300?")
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/27.png)

上述输出很好的展示了ReAct的工作流程：`Thought`->`action`->`action_input`->`Observation`。我们再来看一个调用维基百科API的例子：

````python
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/28.png)

现在我们介绍另一个强大的工具：`PythonREPLTool`。LLM会根据用户需求自动生成python代码，然后交给`PythonREPLTool`去执行。比如下面这个例子：

````python
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
agent.run(f"""Please use Python code (executed via the python_repl tool) to sort the following list of customers by last name, then first name, and print the result: {customer_list}""")
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/29.png)

在输出的详细信息中，我们甚至可以看到生成的python代码。此处，我们也可以设置`langchain.debug=True`来查看更加详细的debug信息。`initialize_agent`是通用的Agent构造器，而`create_python_agent`是专为python代码执行优化的Agent封装。

此外，我们也可以自定义工具：

````python
from langchain.agents import tool
from datetime import date
@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")
````

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/langchain-for-llm-application-development/30.png)

# 8.Conclusion

不再赘述。
