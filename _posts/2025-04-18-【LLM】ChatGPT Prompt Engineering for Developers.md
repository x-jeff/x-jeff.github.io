---
layout:     post
title:      【LLM】ChatGPT Prompt Engineering for Developers
subtitle:   面向开发者的提示工程
date:       2025-04-18
author:     x-jeff
header-img: blogimg/20220511.jpg
catalog: true
tags:
    - Large Language Models
---
>本文为参考吴恩达老师的"ChatGPT Prompt Engineering for Developers"课程所作的个人笔记。
>
>课程地址：[https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)。
>
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Introduction

LLM模型通常分为两类：

* 基础的LLM（Base LLM）：根据输入文本，一个接一个的预测下一个词。
* 经过指令fine-tune过的LLM（Instruction Tuned LLM）：可以根据输入的指令，生成相应的答案。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/chatgpt-prompt-engineering-for-developers/1.png)

# 2.Guidelines

使用ChatGPT的两大原则：

1. 编写清晰且具体的指令。
    * 使用分隔符。
    * 要求结构化的输出。
    * 要求模型检查条件是否被满足。
    * 给少量的提示。
2. 给模型一些时间来思考。
    * 明确完成任务所需的步骤。
    * 让模型自己解决问题。

## 2.1.编写清晰且具体的指令

### 2.1.1.使用分隔符

可以使用`"""`、`\`\`\``、`---`、`<>`、`<tag> <\tag>`等分隔符对不同的指令或上下文进行分隔。实际上，分隔符可以是任何符号，只要这个符号能让模型清楚地知道这是一个单独的部分即可。

````python
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo",temperature=0): 
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
Providing clear and specific instructions to a model is essential for guiding it towards the desired output and reducing the chances of irrelevant or incorrect responses, with longer prompts often providing more clarity and context for more detailed and relevant outputs.
```

在上述代码示例中，我们使用` ``` `将指令和输入文本分隔开来。使用分隔符可以避免**提示词注入（prompt injection）**。在下图所示的例子中，如果我们没有使用分隔符，此时要总结的文本中出现了“忘掉之前的指令”等类似的表达，这就会对我们给出的指令造成干扰，这就是提示词注入问题。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/chatgpt-prompt-engineering-for-developers/2.png)

### 2.1.2.要求结构化的输出

请求一个结构化的输出，比如HTML或JSON格式，可能会有助于模型更容易地输出。比如如下代码示例，我们让模型以JSON的格式生成3本书籍的信息：

````python
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
[
    {
        "book_id": 1,
        "title": "The Midnight Garden",
        "author": "Elena Rivers",
        "genre": "Fantasy"
    },
    {
        "book_id": 2,
        "title": "Echoes of the Past",
        "author": "Nathan Black",
        "genre": "Mystery"
    },
    {
        "book_id": 3,
        "title": "Whispers in the Wind",
        "author": "Samantha Reed",
        "genre": "Romance"
    }
]
```

### 2.1.3.要求模型检查条件是否被满足

比如如下代码示例，我们给了一段关于泡茶流程的文本，我们让模型自己去判断这段文本中是否列出了一系列步骤，如果有，就按照特定的格式将这些步骤输出出来，如果没有，就输出这段文本未提供步骤。

````python
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 1:")
print(response)
````

输出为：

```
Completion for Text 1:
Step 1 - Get some water boiling.
Step 2 - Grab a cup and put a tea bag in it.
Step 3 - Pour the hot water over the tea bag.
Step 4 - Let the tea steep for a few minutes.
Step 5 - Remove the tea bag.
Step 6 - Add sugar or milk to taste.
Step 7 - Enjoy your delicious cup of tea.
```

现在我们将文本换成一段描述一天生活的文字，并且使用相同的提示词，我们来看下模型的输出：

````python
text_2 = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \ 
walk in the park. The flowers are blooming, and the \ 
trees are swaying gently in the breeze. People \ 
are out and about, enjoying the lovely weather. \ 
Some are having picnics, while others are playing \ 
games or simply relaxing on the grass. It's a \ 
perfect day to spend time outdoors and appreciate the \ 
beauty of nature.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_2}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 2:")
print(response)
````

输出为：

```
Completion for Text 2:
No steps provided.
```

### 2.1.4.给少量的提示

给少量的提示（**few-shot prompting**）指的是在提示词中给出一些标准的输入输出样例，让模型去仿照这些样例去输出。比如如下代码示例：

````python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
<grandparent>: The tallest trees weather the strongest storms; the brightest stars shine in the darkest nights; the strongest hearts endure the greatest trials.
```

## 2.2.给模型一些时间来思考

### 2.2.1.明确完成任务所需的步骤

````python
text = f"""
In a charming village, siblings Jack and Jill set out on \ 
a quest to fetch water from a hilltop \ 
well. As they climbed, singing joyfully, misfortune \ 
struck—Jack tripped on a stone and tumbled \ 
down the hill, with Jill following suit. \ 
Though slightly battered, the pair returned home to \ 
comforting embraces. Despite the mishap, \ 
their adventurous spirits remained undimmed, and they \ 
continued exploring with delight.
"""
# example 1
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""
response = get_completion(prompt_1)
print("Completion for prompt 1:")
print(response)
````

输出为：

```
Completion for prompt 1:
1 - Jack and Jill, siblings from a charming village, go on a quest to fetch water from a hilltop well, but encounter misfortune along the way.

2 - Jack et Jill, frère et sœur d'un charmant village, partent en quête d'eau d'un puits au sommet d'une colline, mais rencontrent des malheurs en chemin.

3 - Jack, Jill

4 - 
{
  "french_summary": "Jack et Jill, frère et sœur d'un charmant village, partent en quête d'eau d'un puits au sommet d'une colline, mais rencontrent des malheurs en chemin.",
  "num_names": 2
}
```

我们可以进一步的让输出更加格式化：

````python
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nCompletion for prompt 2:")
print(response)
````

输出为：

```
Completion for prompt 2:
Summary: Jack and Jill, siblings, go on a quest to fetch water from a hilltop well, but encounter misfortune along the way.

Translation: Jack et Jill, frère et sœur, partent en quête d'eau d'un puits au sommet d'une colline, mais rencontrent des malheurs en chemin.

Names: Jack, Jill

Output JSON: 
{
  "french_summary": "Jack et Jill, frère et sœur, partent en quête d'eau d'un puits au sommet d'une colline, mais rencontrent des malheurs en chemin.",
  "num_names": 2
}
```

### 2.2.2.让模型自己解决问题

在匆忙得出结论之前，让模型自己解决问题。我们来看一个例子：

````python
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need \
 help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
response = get_completion(prompt)
print(response)
````

在这个例子中，我们让学生计算建造一个太阳能发电站的费用，这里学生给出的答案是错误的。我们尝试让模型给出它的判断，模型输出见下：

```
The student's solution is correct. The total cost for the first year of operations as a function of the number of square feet is indeed 450x + 100,000.
```

很明显，模型给出了错误的判断。为了解决这个问题，我们可以让模型自己先解出答案，然后将其与学生的答案进行比较，见如下代码：

````python
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem including the final total. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
I'm building a solar power installation and I need help \
working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
``` 
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:
"""
response = get_completion(prompt)
print(response)
````

输出为：

````
Let x be the size of the installation in square feet.

Costs:
1. Land cost: $100 * x
2. Solar panel cost: $250 * x
3. Maintenance cost: $100,000 + $10 * x

Total cost: $100 * x + $250 * x + $100,000 + $10 * x = $360 * x + $100,000
```
The total cost for the first year of operations as a function of the number of square feet is $360 * x + $100,000.
```
Is the student's solution the same as actual solution just calculated:
```
Yes
```
Student grade:
```
correct
```
````

这里我们可以看到模型给出了正确的答案，但是依旧判断错了学生的答案，这部分代码在课程演示的时候是可以正确判断出学生的答案是错误的，但是我自己运行却依旧得出了错误的判断。大家可以试试这段代码，欢迎在留言区讨论。下图是课程演示时老师得到的输出：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/chatgpt-prompt-engineering-for-developers/3.png)

## 2.3.模型的局限性

在面对一些比较晦涩难懂的问题时，模型可能会编造一些听起来合理但实际不正确的答案。我们将这些模型虚构出来的观点称为**幻觉（hallucination）**。下面是模型产生幻觉的一个例子，我们让其描述一个根本不存在的产品，但模型依旧给出了像模像样的答案。

````python
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
The AeroGlide UltraSlim Smart Toothbrush by Boie is a high-tech toothbrush designed to provide a superior cleaning experience. It features ultra-soft bristles that are gentle on the gums and teeth, while still effectively removing plaque and debris. The toothbrush also has a slim design that makes it easy to maneuver and reach all areas of the mouth.

One of the standout features of the AeroGlide UltraSlim Smart Toothbrush is its smart technology. It connects to a mobile app that tracks your brushing habits and provides personalized recommendations for improving your oral hygiene routine. The app also includes a timer to ensure you are brushing for the recommended two minutes.

The toothbrush is made from durable, antimicrobial materials that resist bacteria growth and can be easily cleaned and sanitized. It is also eco-friendly, as the brush head is replaceable and the handle is made from recyclable materials.

Overall, the AeroGlide UltraSlim Smart Toothbrush by Boie is a sleek and innovative toothbrush that offers a thorough and personalized cleaning experience for users.
```

减少幻觉的一个策略是先让模型寻找与问题相关的信息，然后再基于这些相关信息来回答这个问题。

# 3.Iterative

我们可以通过不断迭代优化提示词以最终获得我们想要的答案。我们通过一个例子来说明，下面是一段关于椅子的产品参数：

````python
fact_sheet_chair = """
OVERVIEW
- Part of a beautiful family of mid-century inspired office furniture, 
including filing cabinets, desks, bookcases, meeting tables, and more.
- Several options of shell color and base finishes.
- Available with plastic back and front upholstery (SWC-100) 
or full upholstery (SWC-110) in 10 fabric and 6 leather options.
- Base finish options are: stainless steel, matte black, 
gloss white, or chrome.
- Chair is available with or without armrests.
- Suitable for home or business settings.
- Qualified for contract use.

CONSTRUCTION
- 5-wheel plastic coated aluminum base.
- Pneumatic chair adjust for easy raise/lower action.

DIMENSIONS
- WIDTH 53 CM | 20.87”
- DEPTH 51 CM | 20.08”
- HEIGHT 80 CM | 31.50”
- SEAT HEIGHT 44 CM | 17.32”
- SEAT DEPTH 41 CM | 16.14”

OPTIONS
- Soft or hard-floor caster options.
- Two choices of seat foam densities: 
 medium (1.8 lb/ft3) or high (2.8 lb/ft3)
- Armless or 8 position PU armrests 

MATERIALS
SHELL BASE GLIDER
- Cast Aluminum with modified nylon PA6/PA66 coating.
- Shell thickness: 10 mm.
SEAT
- HD36 foam

COUNTRY OF ORIGIN
- Italy
"""
````

现在我们让模型帮助市场团队撰写一个营销文案：

````python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
Introducing our stylish and versatile mid-century inspired office chair, perfect for both home and business settings. This chair is part of a beautiful family of office furniture that includes filing cabinets, desks, bookcases, meeting tables, and more.

Customize your chair with several options of shell color and base finishes to suit your personal style. Choose between plastic back and front upholstery or full upholstery in a variety of fabric and leather options. The base finish options include stainless steel, matte black, gloss white, or chrome. You can also choose to have armrests or go for a sleek armless design.

Constructed with a 5-wheel plastic coated aluminum base, this chair features a pneumatic chair adjust for easy raise/lower action. The dimensions of the chair are as follows: width 53 cm, depth 51 cm, height 80 cm, seat height 44 cm, and seat depth 41 cm.

Personalize your chair even further with options such as soft or hard-floor caster options, two choices of seat foam densities (medium or high), and the choice between armless or 8 position PU armrests.

Made with high-quality materials, including a cast aluminum shell with modified nylon coating and HD36 foam seat, this chair is designed for durability and comfort. Plus, it is qualified for contract use, making it a reliable choice for any workspace.

Add a touch of Italian elegance to your office with this stylish and functional office chair. Elevate your workspace with our mid-century inspired chair that combines style, comfort, and functionality seamlessly.
```

但这个文案太长了，可能并不是我们想要的，因此，我们可以优化提示词，将输出限制在50个单词以内：

````python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
Introducing our versatile and stylish mid-century office chair, available in a range of colors and finishes. With adjustable height and comfortable upholstery options, this chair is perfect for both home and business use. Made in Italy with quality materials, it's a perfect blend of form and function.
```

模型按照要求一共输出了47个单词。我们可以看到输出的文案中并没有过多的技术参数，如果我们的文案是面向家具商家的，我们可能希望文案中涉及更多的技术参数，并在文案结尾附上产品ID。因此我们可以继续优化提示词：

````python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

At the end of the description, include every 7-character 
Product ID in the technical specification.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
Introducing our versatile and stylish office chair, part of a mid-century inspired furniture collection. Choose from a variety of shell colors and base finishes to suit your space. Constructed with a durable aluminum base and high-density foam seat for comfort. Perfect for home or business use. 

Product IDs: SWC-100, SWC-110
```

我们可以进一步将输出格式化：

````python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

At the end of the description, include every 7-character 
Product ID in the technical specification.

After the description, include a table that gives the 
product's dimensions. The table should have two columns.
In the first column include the name of the dimension. 
In the second column include the measurements in inches only.

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website. 
Place the description in a <div> element.

Technical specifications: ```{fact_sheet_chair}```
"""

response = get_completion(prompt)
print(response)
````

输出为：

```html
<div>
  <p>This mid-century inspired office chair is a perfect addition to any home or business setting. Constructed with a 5-wheel plastic coated aluminum base and a pneumatic chair adjust for easy raise/lower action, this chair offers both style and functionality. The chair is available in various shell colors and base finishes, with options for plastic or full upholstery in a variety of fabric and leather choices. Choose between armless or armrest options to suit your preference. Made with high-quality materials, including a cast aluminum shell with modified nylon coating and HD36 foam seat, this chair is durable and comfortable. Qualified for contract use, this chair is a versatile and stylish choice for any workspace.</p>
  
  <p>Product IDs: SWC-100, SWC-110</p>
  
  <h2>Product Dimensions</h2>
  <table>
    <tr>
      <td>WIDTH</td>
      <td>20.87"</td>
    </tr>
    <tr>
      <td>DEPTH</td>
      <td>20.08"</td>
    </tr>
    <tr>
      <td>HEIGHT</td>
      <td>31.50"</td>
    </tr>
    <tr>
      <td>SEAT HEIGHT</td>
      <td>17.32"</td>
    </tr>
    <tr>
      <td>SEAT DEPTH</td>
      <td>16.14"</td>
    </tr>
  </table>
</div>
```

最后我们可以尝试将这个HTML格式的输出可视化：

```python
from IPython.display import display, HTML
display(HTML(response))
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/chatgpt-prompt-engineering-for-developers/4.png)

# 4.Summarizing

下面是一个文本总结的例子。我们让模型对产品购买者的评论进行总结，并且限制在30个词以内：

````python
prod_review = """
Got this panda plush toy for my daughter's birthday, \
who loves it and takes it everywhere. It's soft and \ 
super cute, and its face has a friendly look. It's \ 
a bit small for what I paid though. I think there \ 
might be other options that are bigger for the \ 
same price. It arrived a day earlier than expected, \ 
so I got to play with it myself before I gave it \ 
to her.
"""

prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words. 

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)
````

输出为：

```
Soft, cute panda plush toy loved by daughter, but smaller than expected for the price. Arrived early, friendly face.
```

如果我们打算将评论总结交给商品运输部门去分析并优化工作，我们可以修改提示词，让总结更关注商品运输的信息：

````python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
Shipping deparmtment. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words, and focusing on any aspects \
that mention shipping and delivery of the product. 

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)
````

输出为：

```
The customer was pleased with the early delivery of the panda plush toy, but felt it was slightly small for the price paid.
```

我们也可以让模型只是提取信息而不是总结：

````python
prompt = f"""
Your task is to extract relevant information from \ 
a product review from an ecommerce site to give \
feedback to the Shipping department. 

From the review below, delimited by triple quotes \
extract the information relevant to shipping and \ 
delivery. Limit to 30 words. 

Review: ```{prod_review}```
"""

response = get_completion(prompt)
print(response)
````

输出为：

```
Feedback: The product arrived a day earlier than expected, which was a pleasant surprise. Customers may prefer larger options for the same price.
```

还可以对多条评论进行批量总结：

````python
review_1 = prod_review 

# review for a standing lamp
review_2 = """
Needed a nice lamp for my bedroom, and this one \
had additional storage and not too high of a price \
point. Got it fast - arrived in 2 days. The string \
to the lamp broke during the transit and the company \
happily sent over a new one. Came within a few days \
as well. It was easy to put together. Then I had a \
missing part, so I contacted their support and they \
very quickly got me the missing piece! Seems to me \
to be a great company that cares about their customers \
and products. 
"""

# review for an electric toothbrush
review_3 = """
My dental hygienist recommended an electric toothbrush, \
which is why I got this. The battery life seems to be \
pretty impressive so far. After initial charging and \
leaving the charger plugged in for the first week to \
condition the battery, I've unplugged the charger and \
been using it for twice daily brushing for the last \
3 weeks all on the same charge. But the toothbrush head \
is too small. I’ve seen baby toothbrushes bigger than \
this one. I wish the head was bigger with different \
length bristles to get between teeth better because \
this one doesn’t.  Overall if you can get this one \
around the $50 mark, it's a good deal. The manufactuer's \
replacements heads are pretty expensive, but you can \
get generic ones that're more reasonably priced. This \
toothbrush makes me feel like I've been to the dentist \
every day. My teeth feel sparkly clean! 
"""

# review for a blender
review_4 = """
So, they still had the 17 piece system on seasonal \
sale for around $49 in the month of November, about \
half off, but for some reason (call it price gouging) \
around the second week of December the prices all went \
up to about anywhere from between $70-$89 for the same \
system. And the 11 piece system went up around $10 or \
so in price also from the earlier sale price of $29. \
So it looks okay, but if you look at the base, the part \
where the blade locks into place doesn’t look as good \
as in previous editions from a few years ago, but I \
plan to be very gentle with it (example, I crush \
very hard items like beans, ice, rice, etc. in the \ 
blender first then pulverize them in the serving size \
I want in the blender then switch to the whipping \
blade for a finer flour, and use the cross cutting blade \
first when making smoothies, then use the flat blade \
if I need them finer/less pulpy). Special tip when making \
smoothies, finely cut and freeze the fruits and \
vegetables (if using spinach-lightly stew soften the \ 
spinach then freeze until ready for use-and if making \
sorbet, use a small to medium sized food processor) \ 
that you plan to use that way you can avoid adding so \
much ice if at all-when making your smoothie. \
After about a year, the motor was making a funny noise. \
I called customer service but the warranty expired \
already, so I had to buy another one. FYI: The overall \
quality has gone done in these types of products, so \
they are kind of counting on brand recognition and \
consumer loyalty to maintain sales. Got it in about \
two days.
"""

reviews = [review_1, review_2, review_3, review_4]

for i in range(len(reviews)):
    prompt = f"""
    Your task is to generate a short summary of a product \ 
    review from an ecommerce site. 

    Summarize the review below, delimited by triple \
    backticks in at most 20 words. 

    Review: ```{reviews[i]}```
    """

    response = get_completion(prompt)
    print(i, response, "\n")
````

输出为：

```
0 Summary: 
Adorable panda plush loved by daughter, but small for price. Arrived early, soft and cute. 

1 Great lamp with storage, fast delivery, excellent customer service for missing parts. Company cares about customers. 

2 Impressive battery life, small brush head, good deal for $50, generic replacement heads available, leaves teeth feeling clean. 

3 17-piece system on sale for $49, prices increased later. Base quality not as good, motor issues after a year. 
```

# 5.Inferring

我们可以让模型对文本进行情感分类：

````python
lamp_review = """
Needed a nice lamp for my bedroom, and this one had \
additional storage and not too high of a price point. \
Got it fast.  The string to our lamp broke during the \
transit and the company happily sent over a new one. \
Came within a few days as well. It was easy to put \
together.  I had a missing part, so I contacted their \
support and they very quickly got me the missing piece! \
Lumina seems to me to be a great company that cares \
about their customers and products!!
"""

prompt = f"""
What is the sentiment of the following product review, 
which is delimited with triple backticks?

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
The sentiment of the review is positive. The reviewer is satisfied with the lamp they purchased, mentioning the additional storage, reasonable price, fast delivery, good customer service, and ease of assembly. They also praise the company for caring about their customers and products.
```

其他一些提示词例子：

````python
#仅回答positive或negative
prompt = f"""
What is the sentiment of the following product review, 
which is delimited with triple backticks?

Give your answer as a single word, either "positive" \
or "negative".

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response) #输出为：Positive

#用不超过5个词描述用户的情感
prompt = f"""
Identify a list of emotions that the writer of the \
following review is expressing. Include no more than \
five items in the list. Format your answer as a list of \
lower-case words separated by commas.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response) #输出为：happy, satisfied, grateful, impressed, content

#用yes或no回答用户是否对商品感到生气
prompt = f"""
Is the writer of the following review expressing anger?\
The review is delimited with triple backticks. \
Give your answer as either yes or no.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response) #输出为：No
````

我们也可以让模型提取产品以及制造公司的名字，并以JSON格式输出：

````python
prompt = f"""
Identify the following items from the review text: 
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Item" and "Brand" as the keys. 
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
  
Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
{
  "Item": "lamp",
  "Brand": "Lumina"
}
```

我们可以将上述提示词合并为一个：

````python
prompt = f"""
Identify the following items from the review text: 
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Sentiment", "Anger", "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
Format the Anger value as a boolean.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
{
    "Sentiment": "positive",
    "Anger": false,
    "Item": "lamp",
    "Brand": "Lumina"
}
```

另一个例子，让模型总结文本所讨论内容的5个主题：

````python
story = """
In a recent survey conducted by the government, 
public sector employees were asked to rate their level 
of satisfaction with the department they work at. 
The results revealed that NASA was the most popular 
department with a satisfaction rating of 95%.

One NASA employee, John Smith, commented on the findings, 
stating, "I'm not surprised that NASA came out on top. 
It's a great place to work with amazing people and 
incredible opportunities. I'm proud to be a part of 
such an innovative organization."

The results were also welcomed by NASA's management team, 
with Director Tom Johnson stating, "We are thrilled to 
hear that our employees are satisfied with their work at NASA. 
We have a talented and dedicated team who work tirelessly 
to achieve our goals, and it's fantastic to see that their 
hard work is paying off."

The survey also revealed that the 
Social Security Administration had the lowest satisfaction 
rating, with only 45% of employees indicating they were 
satisfied with their job. The government has pledged to 
address the concerns raised by employees in the survey and 
work towards improving job satisfaction across all departments.
"""

prompt = f"""
Determine five topics that are being discussed in the \
following text, which is delimited by triple backticks.

Make each item one or two words long. 

Format your response as a list of items separated by commas.

Text sample: '''{story}'''
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
1. Survey
2. Job satisfaction
3. NASA
4. Social Security Administration
5. Government pledge
```

我们也可以给定5个主题，让模型去判断输入文本中是否涵盖了这几个主题：

````python
topic_list = [
    "nasa", "local government", "engineering", 
    "employee satisfaction", "federal government"
]

prompt = f"""
Determine whether each item in the following list of \
topics is a topic in the text below, which
is delimited with triple backticks.

Give your answer as follows:
item from the list: 0 or 1

List of topics: {", ".join(topic_list)}

Text sample: '''{story}'''
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
nasa: 1
local government: 0
engineering: 0
employee satisfaction: 1
federal government: 1
```

# 6.Transforming

首先是一个将英语翻译为西班牙语的简单例子：

````python
prompt = f"""
Translate the following English text to Spanish: \ 
```Hi, I would like to order a blender```
"""
response = get_completion(prompt)
print(response) #输出为：Hola, me gustaría ordenar una licuadora.
````

第二个例子，让模型判断是什么语种：

````python
prompt = f"""
Tell me which language this is: 
```Combien coûte le lampadaire?```
"""
response = get_completion(prompt)
print(response) #输出为：This is French.
````

第三个例子，一次性执行多个翻译：

````python
prompt = f"""
Translate the following  text to French and Spanish
and English pirate: \
```I want to order a basketball```
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
French: Je veux commander un ballon de basket
Spanish: Quiero ordenar un balón de baloncesto
English: I want to order a basketball
```

第四个例子，将一句英语分别翻译为正式的和非正式的西班牙语：

````python
prompt = f"""
Translate the following text to Spanish in both the \
formal and informal forms: 
'Would you like to order a pillow?'
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
Formal: ¿Le gustaría ordenar una almohada?
Informal: ¿Te gustaría ordenar una almohada?
```

第五个例子，批量处理不同的语言：

````python
user_messages = [
  "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal         
  "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
  "Il mio mouse non funziona",                                 # My mouse is not working
  "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
  "我的屏幕在闪烁"                                               # My screen is flashing
] 

for issue in user_messages:
    prompt = f"Tell me what language this is: ```{issue}```"
    lang = get_completion(prompt)
    print(f"Original message ({lang}): {issue}")

    prompt = f"""
    Translate the following  text to English \
    and Korean: ```{issue}```
    """
    response = get_completion(prompt)
    print(response, "\n")
````

输出为：

```
Original message (This is French.): La performance du système est plus lente que d'habitude.
English: "The system performance is slower than usual."

Korean: "시스템 성능이 평소보다 느립니다." 

Original message (This is Spanish.): Mi monitor tiene píxeles que no se iluminan.
English: "My monitor has pixels that do not light up."

Korean: "내 모니터에는 빛나지 않는 픽셀이 있습니다." 

Original message (Italian): Il mio mouse non funziona
English: My mouse is not working
Korean: 내 마우스가 작동하지 않습니다 

Original message (Polish): Mój klawisz Ctrl jest zepsuty
English: My Ctrl key is broken
Korean: 제 Ctrl 키가 고장 났어요 

Original message (This is Chinese.): 我的屏幕在闪烁
English: My screen is flickering
Korean: 내 화면이 깜박거립니다 
```

第六个例子是语气转换，比如将俚语转换为商务信件：

````python
prompt = f"""
Translate the following from slang to a business letter: 
'Dude, This is Joe, check out this spec on this standing lamp.'
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
Dear Sir/Madam,

I am writing to bring to your attention the specifications of a standing lamp that I believe may be of interest to you. 

Sincerely,
Joe
```

第七个例子是格式转换，比如将JSON转换为HTML：

````python
data_json = { "resturant employees" :[ 
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json}
"""
response = get_completion(prompt)
print(response)
````

输出为：

```
<html>
<head>
  <title>Restaurant Employees</title>
</head>
<body>
  <table>
    <tr>
      <th>Name</th>
      <th>Email</th>
    </tr>
    <tr>
      <td>Shyam</td>
      <td>shyamjaiswal@gmail.com</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>bob32@gmail.com</td>
    </tr>
    <tr>
      <td>Jai</td>
      <td>jai87@gmail.com</td>
    </tr>
  </table>
</body>
</html>
```

可视化生成的html：

```python
from IPython.display import display, Markdown, Latex, HTML, JSON
display(HTML(response))
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/chatgpt-prompt-engineering-for-developers/5.png)

第八个例子是拼写检查和语法检查：

````python
text = [ 
  "The girl with the black and white puppies have a ball.",  # The girl has a ball.
  "Yolanda has her notebook.", # ok
  "Its going to be a long day. Does the car need it’s oil changed?",  # Homonyms
  "Their goes my freedom. There going to bring they’re suitcases.",  # Homonyms
  "Your going to need you’re notebook.",  # Homonyms
  "That medicine effects my ability to sleep. Have you heard of the butterfly affect?", # Homonyms
  "This phrase is to cherck chatGPT for speling abilitty"  # spelling
]
for t in text:
    prompt = f"""Proofread and correct the following text
    and rewrite the corrected version. If you don't find
    and errors, just say "No errors found". Don't use 
    any punctuation around the text:
    ```{t}```"""
    response = get_completion(prompt)
    print(response)
````

输出为：

```
The girl with the black and white puppies has a ball.
No errors found.
It's going to be a long day. Does the car need its oil changed?
There goes my freedom. They're going to bring their suitcases.
You're going to need your notebook.
That medicine affects my ability to sleep. Have you heard of the butterfly effect?
This phrase is to check chatGPT for spelling ability.
```

# 7.Expanding

文本扩展是LLM的一个重要功能，下面的例子是让LLM充当AI客服经理，根据客户的反馈来回复客户的邮件：

````python
# given the sentiment from the lesson on "inferring",
# and the original customer message, customize the email
sentiment = "negative"

# review for a blender
review = f"""
So, they still had the 17 piece system on seasonal \
sale for around $49 in the month of November, about \
half off, but for some reason (call it price gouging) \
around the second week of December the prices all went \
up to about anywhere from between $70-$89 for the same \
system. And the 11 piece system went up around $10 or \
so in price also from the earlier sale price of $29. \
So it looks okay, but if you look at the base, the part \
where the blade locks into place doesn’t look as good \
as in previous editions from a few years ago, but I \
plan to be very gentle with it (example, I crush \
very hard items like beans, ice, rice, etc. in the \ 
blender first then pulverize them in the serving size \
I want in the blender then switch to the whipping \
blade for a finer flour, and use the cross cutting blade \
first when making smoothies, then use the flat blade \
if I need them finer/less pulpy). Special tip when making \
smoothies, finely cut and freeze the fruits and \
vegetables (if using spinach-lightly stew soften the \ 
spinach then freeze until ready for use-and if making \
sorbet, use a small to medium sized food processor) \ 
that you plan to use that way you can avoid adding so \
much ice if at all-when making your smoothie. \
After about a year, the motor was making a funny noise. \
I called customer service but the warranty expired \
already, so I had to buy another one. FYI: The overall \
quality has gone done in these types of products, so \
they are kind of counting on brand recognition and \
consumer loyalty to maintain sales. Got it in about \
two days.
"""

prompt = f"""
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer.
Given the customer email delimited by ```, \
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for \
their review.
If the sentiment is negative, apologize and suggest that \
they can reach out to customer service. 
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as `AI customer agent`.
Customer review: ```{review}```
Review sentiment: {sentiment}
"""
response = get_completion(prompt, temperature=0.7)
print(response)
````

输出为：

```
Dear valued customer,

Thank you for sharing your detailed feedback with us regarding your recent purchase. We are sorry to hear about the issues you have experienced with the pricing changes, product quality, and the motor noise after a year of use. We apologize for any inconvenience this may have caused you.

If you have any further concerns or would like to discuss this matter further, please feel free to reach out to our customer service team. We are here to assist you in any way we can.

Thank you for your loyalty and for taking the time to provide us with your feedback.

AI customer agent
```

我们可以通过[temperature](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#41temperature-scaling)参数控制输出的随机性。

# 8.Chatbot

我们重新定义一个函数，让我们可以设置多种`role`：

````python
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

messages =  [  
{'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},    
{'role':'user', 'content':'tell me a joke'},   
{'role':'assistant', 'content':'Why did the chicken cross the road'},   
{'role':'user', 'content':'I don\'t know'}  ]
````

`role`一共有3种：

1. `system`：是一个整体指导，告诉LLM它的角色是什么。比如这个例子中，我们告诉模型它是一个说话风格像莎士比亚的助手。
2. `user`：是用户的输入。比如这个例子中，我们让模型讲一个笑话。
3. `assistant`：是ChatGPT的回复消息。比如这个例子中，ChatGPT回复“鸡为什么过马路”。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/chatgpt-prompt-engineering-for-developers/6.png)

````python
response = get_completion_from_messages(messages, temperature=1)
print(response)
````

输出为：

```
To get to the other side, perchance! A jest both old and oft heard, but a merry one nonetheless.
```

另一个例子：

````python
messages =  [  
{'role':'system', 'content':'You are friendly chatbot.'},    
{'role':'user', 'content':'Hi, my name is Isa'}  ]
response = get_completion_from_messages(messages, temperature=1)
print(response)
````

输出为：

```
Hello Isa! Nice to meet you. How are you doing today?
```

我们再次向模型询问自己的名字是什么：

````python
messages =  [  
{'role':'system', 'content':'You are friendly chatbot.'},    
{'role':'user', 'content':'Yes,  can you remind me, What is my name?'}  ]
response = get_completion_from_messages(messages, temperature=1)
print(response)
````

输出为：

```
I'm sorry, but I don't have access to your name or any personal information about you. How can I assist you today?
```

我们已经在前一个对话中告诉了模型自己的名字，但这个对话中，模型依旧不知道自己的名字，这是因为每个对话都是独立的，我们可以通过给模型上下文来解决这个问题。

````python
messages =  [  
{'role':'system', 'content':'You are friendly chatbot.'},
{'role':'user', 'content':'Hi, my name is Isa'},
{'role':'assistant', 'content': "Hi Isa! It's nice to meet you. \
Is there anything I can help you with today?"},
{'role':'user', 'content':'Yes, you can remind me, What is my name?'}  ]
response = get_completion_from_messages(messages, temperature=1)
print(response)
````

输出为：

```
Your name is Isa.
```

现在我们来为披萨店搭建一个自动订餐机器人：

````python
def collect_messages(_):
    #inp的定义见下
    #用户在输入框键入的内容作为提示词
    prompt = inp.value_input
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    #增加上下文信息
    context.append({'role':'assistant', 'content':f"{response}"})
    #前端界面的展示格式
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))
 
    return pn.Column(*panels)

import panel as pn  # GUI
pn.extension()

panels = [] # collect display 

#role为system，定义了LLM的角色
#content部分翻译如下：
# 你是 OrderBot，一名自动化的披萨餐厅点单助手。
# 你会先向顾客打招呼，然后开始接收订单，
# 接着询问是自取还是外送。
# 你会等待用户完整下完订单，再进行汇总，
# 并最后确认一次是否还要添加其他内容。
# 如果是外送，你还要收集送餐地址。
# 最后，你将收款。
# 请确保对菜单中的所有选项、加料和尺寸进行澄清，
# 以便准确识别顾客所点的具体商品。
# 你的语气应简短、轻松、友好、有对话感。
# 菜单如下：
# 披萨类：
# 意大利香肠披萨：12.95、10.00、7.00
# 奶酪披萨：10.95、9.25、6.50
# 茄子披萨：11.95、9.75、6.75
# 小食：
# 薯条：4.50、3.50
# 希腊沙拉：7.25
# 配料：
# 加奶酪：2.00
# 蘑菇：1.50
# 香肠：3.00
# 加拿大培根：3.50
# AI 酱：1.50
# 青椒：1.00
# 饮品：
# 可乐：3.00、2.00、1.00
# 雪碧：3.00、2.00、1.00
# 瓶装水：5.00
context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to collect orders for a pizza restaurant. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza  12.95, 10.00, 7.00 \
cheese pizza   10.95, 9.25, 6.50 \
eggplant pizza   11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""} ]  # accumulate messages

#定义单行文本输入框
inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
#点击这个按钮会触发对话
button_conversation = pn.widgets.Button(name="Chat!")

#当按钮被点击时，就会调用collect_messages函数
interactive_conversation = pn.bind(collect_messages, button_conversation)

#定义页面布局
#整体是垂直排列：输入框、按钮、聊天内容展示区域
dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

#展示界面
dashboard
````

聊天界面如下：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMCourse/chatgpt-prompt-engineering-for-developers/7.png)

我们让模型将订单信息转换为JSON格式，方便我们发送给订单系统：

````python
messages =  context.copy()
messages.append(
{'role':'system', 'content':'create a json summary of the previous food order. Itemize the price for each item\
 The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},    
)  

#使用较低的temperature，因为这种场景我们希望输出结果更稳定一些
response = get_completion_from_messages(messages, temperature=0)
print(response)
````

输出为：

```json
{
    "order": {
        "pizza": {
            "item": "Medium Eggplant Pizza",
            "price": 9.75
        },
        "toppings": [],
        "drinks": [],
        "sides": {
            "item": "Fries",
            "price": 3.50
        },
        "total_price": 13.25
    }
}
```
