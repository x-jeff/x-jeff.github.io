---
layout:     post
title:      【从零开始构建大语言模型】【7】【Fine-tuning to follow instructions】
subtitle:   Introduction to instruction fine-tuning，Preparing a dataset for supervised instruction fine-tuning，Organizing data into training batches，Creating data loaders for an instruction dataset，Loading a pretrained LLM，Fine-tuning the LLM on instruction data，Extracting and saving responses，Evaluating the fine-tuned LLM
date:       2025-03-31
author:     x-jeff
header-img: blogimg/20201017.jpg
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

# 1.Fine-tuning to follow instructions

instruction fine-tuning是开发用于聊天机器人、个人助手以及其他对话类任务的大语言模型的核心技术之一。

![Fig7.1](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/1.png)

# 2.Introduction to instruction fine-tuning

我们现在已经了解，预训练LLM是一个逐词生成的训练过程。经过预训练后，LLM具备了文本补全的能力，也就是说，它可以根据给定的片段完成句子或生成段落。然而，预训练的LLM在应对具体指令方面常常表现不佳，比如“修正这段文字的语法”或“将这段文字改为被动语态”。稍后，我们将通过一个具体示例来探讨如何在预训练模型的基础上进行instruction fine-tuning，这一过程也被称为有监督的instruction fine-tuning。

在这里，我们的重点是提升LLM理解和执行此类指令的能力，并生成符合预期的响应，如Fig7.2所示。数据集的准备是instruction fine-tuning中的关键步骤。接下来，我们将完成instruction fine-tuning流程中的三个阶段的所有步骤，从数据集准备开始，如Fig7.3所示。

![Fig7.2](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/2.png)

![Fig7.3](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/3.png)

# 3.Preparing a dataset for supervised instruction fine-tuning

所要下载的数据集包含了1100条类似于Fig7.2中所示的instruction–response对。数据集采用JSON格式。

```python
#Downloading the dataset
import json
import os
import urllib


def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # The book originally contained this unnecessary "else" clause:
    #else:
    #    with open(file_path, "r", encoding="utf-8") as file:
    #        text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data)) #Number of entries: 1100
```

我们查看下数据集中的两个示例：

```python
print("Example entry:\n", data[50])
print("Another example entry:\n", data[999])
```

输出为：

```
Example entry:
 {'instruction': 'Identify the correct spelling of the following word.', 'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}
Another example entry:
 {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
```

我们可以看到每条数据包含3个部分：`'instruction'`、`'input'`和`'output'`。

instruction fine-tuning的过程是让模型在一个明确提供输入-输出对的数据集上进行训练，这些数据对就像我们从JSON文件中提取的那样。针对LLM，有多种方法可以对这些数据进行格式化。Fig7.4展示了两种不同的示例格式，这些格式通常被称为**提示风格（prompt styles）**，在训练诸如Alpaca和Phi-3等知名LLM时都有使用。

![Fig7.4](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/4.png)

我们将采用Alpaca的提示风格，因为它是最受欢迎的风格之一。

现在我们来定义一个名为`format_input`的函数，用于将数据列表中的条目转换为Alpaca风格的输入格式。

```python
#Implementing the prompt formatting function
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)
```

输出为：

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion.'
```

另一个`'input'`为空的例子：

```python
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"

print(model_input + desired_response)
```

输出为：

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is an antonym of 'complicated'?

### Response:
An antonym of 'complicated' is 'simple'.
```

然后我们将数据集划分为训练集、验证集和测试集。

```python
#Partitioning the dataset
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data)) #Training set length: 935
print("Validation set length:", len(val_data)) #Validation set length: 55
print("Test set length:", len(test_data)) #Test set length: 110
```

# 4.Organizing data into training batches

![Fig7.5](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/5.png)

在[第6章](https://shichaoxin.com/2025/03/21/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-6-Fine-tuning-for-classification/)中，训练batch是由PyTorch的`DataLoader`类自动创建的，该类使用默认的collate函数将样本组合成batch。collate函数的作用是将一个个独立的数据样本合并为一个batch，从而使模型在训练过程中能够高效地处理这些数据。

然而，instruction fine-tuning的batch过程要更复杂一些，这就需要我们自定义一个collate函数，然后会将其传入`DataLoader`中使用。我们之所以要实现这个自定义的collate函数，是为了满足instruction fine-tuning数据集在格式和处理上的特殊需求。

我们将分几个步骤来实现batch过程，包括编写自定义的collate函数，如Fig7.6所示。

![Fig7.6](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/6.png)

首先，为了实现步骤2.1和步骤2.2，我们将编写一个`InstructionDataset`类，它会对数据集中的所有输入应用`format_input`函数并进行预先分词（**pretokenizes**），详见Fig7.7。

![Fig7.7](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/7.png)

```python
#Implementing an instruction dataset class
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
```

然后需要对所有输入进行padding，使其长度一致，我们依旧使用`<|endoftext|>`作为padding token。

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) #[50256]
```

接下来进入Fig7.6中的步骤2.3，我们将开发一个自定义的collate函数。这个自定义的collate函数会对每个batch中的训练样本进行padding，使它们的长度一致，同时允许不同的batch具有不同的最大长度，如Fig7.8所示。这种做法通过仅将序列padding到当前batch中最长的样本长度，从而减少了不必要的padding，提高了效率。

![Fig7.8](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/8.png)

```python
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (the extra padding token will be relevant in later codes)
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

print(custom_collate_draft_1(batch))
```

输出为：

```
tensor([[    0,     1,     2,     3,     4],
        [    5,     6, 50256, 50256, 50256],
        [    7,     8,     9, 50256, 50256]])
```

接下来需要继续对自定义的collate函数进行修改，使其除了返回输入token ID外，还返回目标token ID。

![Fig7.9](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/9.png)

与我们在预训练大语言模型时使用的流程类似，目标token ID与输入token ID相同，但向右偏移一个位置。如Fig7.10所示，这样的设置使得LLM能够学习如何预测序列中的下一个token。

![Fig7.10](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/10.png)

```python
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)
```

输出为：

```
tensor([[    0,     1,     2,     3,     4],
        [    5,     6, 50256, 50256, 50256],
        [    7,     8,     9, 50256, 50256]])
tensor([[    1,     2,     3,     4, 50256],
        [    6, 50256, 50256, 50256, 50256],
        [    8,     9, 50256, 50256, 50256]])
```

接下来，我们将所有的padding token分配一个占位值`-100`，如Fig7.11所示。这个特殊的值可以让我们在计算训练损失时忽略这些padding token，从而确保只有有意义的数据才会影响模型的学习（在进行classification fine-tuning时，我们无需考虑这一点，因为当时只使用了模型的最后一个输出token进行训练）。

![Fig7.11](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/11.png)

不过需要注意的是，我们会在目标列表中保留一个文本结束标记（end-of-text token），其ID为50256，如Fig7.12所示。保留该标记可以让LLM学会在响应指令时何时生成文本结束标记，我们将其作为判断生成响应是否完成的一个标志。

![Fig7.12](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/12.png)

在下面的代码中，我们对自定义的collate函数进行了修改，将目标列表中ID为`50256`的token替换为`-100`。此外，我们还引入了一个可选参数`allowed_max_length`，用于限制样本的最大长度。如果打算使用自己的数据集，而该数据集超过了[GPT-2](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/#2gpt2)模型支持的1024个token的上下文长度，这一调整将非常有用。

```python
#Implementing a custom batch collate function
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu" #device可以是cpu，cuda（针对NVIDIA GPU），mps（针对Apple Silicon芯片的Mac）
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)
```

输出为：

```
tensor([[    0,     1,     2,     3,     4],
        [    5,     6, 50256, 50256, 50256],
        [    7,     8,     9, 50256, 50256]])
tensor([[    1,     2,     3,     4, 50256],
        [    6, 50256,  -100,  -100,  -100],
        [    8,     9, 50256,  -100,  -100]])
```

通过一个例子来说明使用`-100`的原因。

```python
logits_1 = torch.tensor(
    [[-1.0, 1.0],  # 1st training example
     [-0.5, 1.5]]  # 2nd training example
)
targets_1 = torch.tensor([0, 1])


loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1) #tensor(1.1269)

logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]  # New 3rd training example
)
targets_2 = torch.tensor([0, 1, 1])

loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2) #tensor(0.7936)

targets_3 = torch.tensor([0, 1, -100]) #把targets_2中的第三个值换成-100

loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3) #tensor(1.1269)
print("loss_1 == loss_3:", loss_1 == loss_3) #loss_1 == loss_3: tensor(True)
```

从上述例子可以看到，交叉熵损失函数自动忽略了`targets_3`中值为`-100`的样本。这是因为PyTorch中的交叉熵函数默认设置为`cross_entropy(..., ignore_index=-100)`，也就是说，它会忽略目标中标记为`-100`的位置。

除了屏蔽padding token外，通常还会屏蔽指令部分，如Fig7.13所示。这样，模型的训练重点将放在生成准确的响应上，而不是死记硬背指令内容，从而有助于降低过拟合的风险。

![Fig7.13](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/13.png)

# 5.Creating data loaders for an instruction dataset

![Fig7.14](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/14.png)

我们使用Python标准库`functools`提供的`partial`函数，来固定一个函数的一部分参数，从而创建一个新的函数，这个新函数可以像原函数一样调用，但其中某些参数的值已经预先设置好了。此外，我们还将`allowed_max_length`设置为`1024`，以便将数据截断到[GPT-2](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/#2gpt2)模型所支持的最大上下文长度，这也是我们之后要进行微调的模型：

```python
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)
```

```python
#Initializing the data loaders
from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
```

输出为：

```
Train loader:
torch.Size([8, 61]) torch.Size([8, 61])
torch.Size([8, 76]) torch.Size([8, 76])
torch.Size([8, 73]) torch.Size([8, 73])
torch.Size([8, 68]) torch.Size([8, 68])
torch.Size([8, 65]) torch.Size([8, 65])
torch.Size([8, 72]) torch.Size([8, 72])
torch.Size([8, 80]) torch.Size([8, 80])
torch.Size([8, 67]) torch.Size([8, 67])
torch.Size([8, 62]) torch.Size([8, 62])
torch.Size([8, 75]) torch.Size([8, 75])
torch.Size([8, 62]) torch.Size([8, 62])
torch.Size([8, 68]) torch.Size([8, 68])
torch.Size([8, 67]) torch.Size([8, 67])
torch.Size([8, 77]) torch.Size([8, 77])
torch.Size([8, 69]) torch.Size([8, 69])
torch.Size([8, 79]) torch.Size([8, 79])
torch.Size([8, 71]) torch.Size([8, 71])
torch.Size([8, 66]) torch.Size([8, 66])
torch.Size([8, 83]) torch.Size([8, 83])
torch.Size([8, 68]) torch.Size([8, 68])
torch.Size([8, 80]) torch.Size([8, 80])
torch.Size([8, 71]) torch.Size([8, 71])
torch.Size([8, 69]) torch.Size([8, 69])
torch.Size([8, 65]) torch.Size([8, 65])
...
torch.Size([8, 83]) torch.Size([8, 83])
torch.Size([8, 66]) torch.Size([8, 66])
torch.Size([8, 74]) torch.Size([8, 74])
torch.Size([8, 69]) torch.Size([8, 69])
```

借助于我们自定义的collate函数，数据加载器能够生成长度不同的batch。

```python
print(inputs[0])
print(targets[0])
```

输出为：

```
tensor([21106,   318,   281, 12064,   326,  8477,   257,  4876,    13, 19430,
          257,  2882,   326, 20431, 32543,   262,  2581,    13,   198,   198,
        21017, 46486,    25,   198, 30003,  6525,   262,  6827,  1262,   257,
          985,   576,    13,   198,   198, 21017, 23412,    25,   198,   464,
         5156,   318,   845, 13779,    13,   198,   198, 21017, 18261,    25,
          198,   464,  5156,   318,   355, 13779,   355,   257,  4936,    13,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
       device='cuda:0')
tensor([  318,   281, 12064,   326,  8477,   257,  4876,    13, 19430,   257,
         2882,   326, 20431, 32543,   262,  2581,    13,   198,   198, 21017,
        46486,    25,   198, 30003,  6525,   262,  6827,  1262,   257,   985,
          576,    13,   198,   198, 21017, 23412,    25,   198,   464,  5156,
          318,   845, 13779,    13,   198,   198, 21017, 18261,    25,   198,
          464,  5156,   318,   355, 13779,   355,   257,  4936,    13, 50256,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100],
       device='cuda:0')
```

# 6.Loading a pretrained LLM

在开始instruction fine-tuning之前，我们首先需要加载一个预训练的GPT模型作为fine-tune的基础，如Fig7.15所示。这次我们会加载一个中等规模的模型，参数量为3.55亿。

![Fig7.15](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/15.png)

```python
#Loading the pretrained model
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();
```

`GPTModel`的定义见：[Coding the GPT model](https://shichaoxin.com/2025/03/03/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-4-Implementing-a-GPT-model-from-scratch-to-generate-text/#7coding-the-gpt-model)，`load_weights_into_gpt`的定义见：[Loading pretrained weights from OpenAI](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#6loading-pretrained-weights-from-openai)。

下载过程：

```
checkpoint: 100%|██████████| 77.0/77.0 [00:00<00:00, 38.3kiB/s]
encoder.json: 100%|██████████| 1.04M/1.04M [00:01<00:00, 728kiB/s] 
hparams.json: 100%|██████████| 91.0/91.0 [00:00<00:00, 45.7kiB/s]
model.ckpt.data-00000-of-00001: 100%|██████████| 1.42G/1.42G [02:31<00:00, 9.36MiB/s]  
model.ckpt.index: 100%|██████████| 10.4k/10.4k [00:00<00:00, 5.20MiB/s]
model.ckpt.meta: 100%|██████████| 927k/927k [00:01<00:00, 642kiB/s]  
vocab.bpe: 100%|██████████| 456k/456k [00:00<00:00, 475kiB/s]  
```

现在，让我们花点时间评估一下预训练LLM在某个验证任务上的表现，通过将其输出与预期响应进行对比。这将帮助我们建立一个基线，了解模型在未经fine-tune的情况下，对指令任务的初始执行能力，也能让我们在后续更好地理解fine-tune所带来的改进效果。我们将使用验证集中的第一个样本来进行这项评估：

```python
torch.manual_seed(123)

input_text = format_input(val_data[0])
print(input_text)
```

输出为：

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
```

接下来，我们使用`generate`函数（定义见：[Modifying the text generation function](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#43modifying-the-text-generation-function)）生成响应：

```python
from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
print(generated_text)
```

输出为：

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'

### Response:

The chef cooks the meal every day.

### Instruction:

Convert the active sentence to passive: 'The chef cooks the
```

[`generate`函数](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#43modifying-the-text-generation-function)返回的是输入文本和输出文本的组合。这种行为在之前是很方便的，因为预训练的LLM主要被设计为文本补全模型，它会将输入和输出拼接在一起，生成连贯、易读的文本。然而，当我们想评估模型在特定任务中的表现时，通常更希望单独关注模型生成的响应部分。

为了单独提取模型生成的响应文本，我们需要从`generated_text`的开头去除输入指令部分：

```python
response_text = generated_text[len(input_text):].strip()
print(response_text)
```

输出为：

```
### Response:

The chef cooks the meal every day.

### Instruction:

Convert the active sentence to passive: 'The chef cooks the
```

这个输出表明，预训练模型尚未具备正确执行所给指令的能力。

# 7.Fine-tuning the LLM on instruction data

![Fig7.16](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/16.png)

在开始训练之前，我们先计算下在训练集和验证集上的初始loss：

```python
from previous_chapters import (
    calc_loss_loader,
    train_model_simple
)

model.to(device)

torch.manual_seed(123)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss) #Training loss: 3.825910234451294
print("Validation loss:", val_loss) #Validation loss: 3.7619343757629395
```

`calc_loss_loader`的定义见：[Calculating the training and validation set losses](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#23calculating-the-training-and-validation-set-losses)。

下表是不同的[GPT-2](https://shichaoxin.com/2024/03/20/LLM-%E4%B8%80%E6%96%87%E8%AF%BB%E6%87%82ChatGPT%E8%83%8C%E5%90%8E%E7%9A%84%E6%8A%80%E6%9C%AF/#2gpt2)模型在不同device上的耗时：

![table](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/17.png)

```python
#Instruction fine-tuning the pretrained LLM
import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

`train_model_simple`的定义见：[Training an LLM](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#3training-an-llm)。训练过程：

```
Ep 1 (Step 000000): Train loss 2.637, Val loss 2.626
Ep 1 (Step 000005): Train loss 1.174, Val loss 1.102
Ep 1 (Step 000010): Train loss 0.872, Val loss 0.944
Ep 1 (Step 000015): Train loss 0.857, Val loss 0.906
Ep 1 (Step 000020): Train loss 0.776, Val loss 0.881
Ep 1 (Step 000025): Train loss 0.754, Val loss 0.859
Ep 1 (Step 000030): Train loss 0.799, Val loss 0.836
Ep 1 (Step 000035): Train loss 0.714, Val loss 0.808
Ep 1 (Step 000040): Train loss 0.672, Val loss 0.806
Ep 1 (Step 000045): Train loss 0.633, Val loss 0.789
Ep 1 (Step 000050): Train loss 0.663, Val loss 0.783
Ep 1 (Step 000055): Train loss 0.760, Val loss 0.763
Ep 1 (Step 000060): Train loss 0.719, Val loss 0.743
Ep 1 (Step 000065): Train loss 0.653, Val loss 0.735
Ep 1 (Step 000070): Train loss 0.532, Val loss 0.729
Ep 1 (Step 000075): Train loss 0.569, Val loss 0.728
Ep 1 (Step 000080): Train loss 0.605, Val loss 0.725
Ep 1 (Step 000085): Train loss 0.509, Val loss 0.709
Ep 1 (Step 000090): Train loss 0.562, Val loss 0.691
Ep 1 (Step 000095): Train loss 0.500, Val loss 0.682
Ep 1 (Step 000100): Train loss 0.503, Val loss 0.677
Ep 1 (Step 000105): Train loss 0.564, Val loss 0.670
Ep 1 (Step 000110): Train loss 0.555, Val loss 0.666
Ep 1 (Step 000115): Train loss 0.508, Val loss 0.664
Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The meal is prepared every day by the chef.<|endoftext|>The following is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive:
...
Ep 2 (Step 000225): Train loss 0.347, Val loss 0.661
Ep 2 (Step 000230): Train loss 0.294, Val loss 0.656
Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The meal is cooked every day by the chef.<|endoftext|>The following is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: What is the capital of the United Kingdom
Training completed in 88.43 minutes.
```

>`gpt2-medium (355M)`模型在我自己笔记本（GPU为NVIDIA T1200）上，只训练两个epoch就用了88分钟😓。

训练输出表明模型正在有效地学习，因为在两个epoch中，训练loss和验证loss都在持续下降。这一结果说明模型在理解和执行指令方面的能力正在逐步提升。由于模型在两个epoch内就已展现出良好的学习效果，因此继续训练到第三个epoch或更长时间并不是必要的，甚至可能适得其反，导致过拟合的风险增加。

经历2个epoch的训练之后，模型成功的生成了正确的响应，将句子`"The chef cooks the meal every day."`转换为了被动语态：`"The meal is cooked every day by the chef."`。

现在，让我们看下训练loss和验证loss的变化曲线。

```python
from previous_chapters import plot_losses

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
```

`plot_losses`的定义见：[Training an LLM](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#3training-an-llm)。

![Fig7.17](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/18.png)

# 8.Extracting and saving responses

我们现在可以评估其在测试集上的表现了。首先，我们将为测试集中每条输入提取模型生成的响应，并将其收集起来以便进行人工分析；随后，我们将对LLM的表现进行评估，以量化其生成响应的质量，如Fig7.18所示。

![Fig7.18](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/19.png)

为了完成指令响应的提取，我们使用[`generate`函数](https://shichaoxin.com/2025/03/13/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%9E%84%E5%BB%BA%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-5-Pretraining-on-unlabeled-data/#43modifying-the-text-generation-function)。随后，我们将模型生成的响应与测试集中前三个样本的期望答案并排打印出来，方便进行对比：

```python
torch.manual_seed(123)


for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")
```

输出为：

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Rewrite the sentence using a simile.

### Input:
The car is very fast.

Correct response:
>> The car is as fast as lightning.

Model response:
>> The car is as fast as a cheetah.
-------------------------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What type of cloud is typically associated with thunderstorms?

Correct response:
>> The type of cloud typically associated with thunderstorms is cumulonimbus.

Model response:
>> The type of cloud associated with thunderstorms is a cumulus cloud.
-------------------------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Name the author of 'Pride and Prejudice'.

Correct response:
>> Jane Austen.

Model response:
>> The author of 'Pride and Prejudice' is Jane Austen.
-------------------------------------
```

从上述输出来看，模型的效果还不错。第一条指令和第三条指令，模型都给出了正确的答案。第二条指令，模型给出的答案是积云（cumulus cloud），而正确答案是积雨云（cumulonimbus），模型的预测结果虽然不完全准确，但也很接近，并且值得注意的是，积云确实可以发展成积雨云。

经过instruction fine-tuning后的LLM可以通过以下多种方式进行评估：

* 通过和标准答案比较来评估模型。比如MMLU（Measuring Massive Multitask Language Understanding，[https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)），其涉及多个学科领域，题目形式基本都是选择题（仅有少量简答题），且均配有标准答案。
* 通过人类打分比较多个大模型之间的对话质量。比如LMSYS chatbot arena：[https://lmarena.ai/](https://lmarena.ai/)。
* 用另一个强大的LLM（如GPT-4）来评判多个模型的回答质量。比如AlpacaEval：[https://tatsu-lab.github.io/alpaca_eval/](https://tatsu-lab.github.io/alpaca_eval/)。

在实际应用中，可以同时考虑这三种评估方法。

考虑到当前任务的规模，我们将使用另一个更强大的大语言模型对响应进行自动化评估。这种方法可以高效地评估生成响应的质量，无需大量人工参与，从而节省时间和资源，同时仍能获得具有参考价值的性能指标。

我们将模型生成的答案和之前的测试集一起放在`test_data`中：

```python
#在整个测试集上运行模型，对测试集中的每个样本都生成答案
from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text

#保存到json文件
with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
```

输出为：

```
100%|██████████| 110/110 [09:12<00:00,  5.02s/it] 
```

查看其中的一条记录：

```python
print(test_data[0])
```

输出为：

```
{'instruction': 'Rewrite the sentence using a simile.', 'input': 'The car is very fast.', 'output': 'The car is as fast as lightning.', 'model_response': 'The car is as fast as a cheetah.'}
```

最后，我们将模型保存为`gpt2-medium355M-sft.pth`，方便以后复用：

```python
import re


file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}") #Model saved as gpt2-medium355M-sft.pth

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
```

# 9.Evaluating the fine-tuned LLM

![Fig7.19](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/20.png)

我们使用[Ollama](https://ollama.com/)在本地运行参数量为8B的Llama 3模型，该模型由Meta AI开发。

>Ollama是对开源库[llama.cpp](https://github.com/ggml-org/llama.cpp)的封装，该库使用纯C/C++实现LLM，以最大化运行效率。然而，Ollama仅用于使用LLM生成文本（即推理），不支持训练或fine-tune LLM。

Ollama安装完成之后，我们可以通过启动Ollama应用程序或者在终端输入`ollama serve`来启用Ollama，然后在另一个终端中运行`ollama run llama3`来执行Llama 3模型的下载，如Fig7.20所示。

![Fig7.20](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/21.png)

模型下载过程如下所示：

![Fig](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/22.png)

我们可以测试一下模型是否好用：

![Fig](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/23.png)

模型的加载和运行都没问题。接下来，我们将一直保持`ollama serve`命令或Ollama应用程序处于运行状态，下面的代码用于验证Ollama session是否正常运行：

```python
import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running:", check_if_running("ollama")) #Ollama running: True
```

除了在终端和模型进行交互之外，还可以使用Python通过其REST API与模型进行交互。下述代码中的`query_model`函数演示了如何使用该API。

```python
import urllib.request

def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)
```

输出为：

```
Llamas are herbivores, which means they primarily eat plants and plant-based foods. Their diet typically consists of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses, short grasses, and even weeds.
2. Hay: Hay is a staple in a llama's diet. They enjoy eating timothy hay, alfalfa hay, or other types of hay as a source of fiber and nutrients.
3. Grains: Llamas may be fed grains like oats, barley, or corn as a treat or to supplement their diet.
4. Fruits and vegetables: Fresh fruits and vegetables can be given to llamas as a treat or to add variety to their diet. Some examples include apples, carrots, sweet potatoes, and leafy greens like kale or spinach.
5. Minerals: Llamas need access to minerals like calcium, phosphorus, and salt to stay healthy. These can be provided through mineral blocks or loose minerals.

In the wild, llamas would typically eat whatever plants are available in their habitat, including:

* Leaves
* Twigs
* Bark
* Fruits
* Flowers

Domesticated llamas may have a more controlled diet, but they still require access to a variety of plant-based foods to stay healthy and thrive.
```

现在，我们使用Llama 3模型对测试集生成参考答案，并基于此对我们之前fine-tune模型的输出答案进行评分，评分范围为0到100分。

首先，我们看下测试集中前三个样本的结果：

```python
for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt)) #给预测输出和GT，直接让Llama 3打分
    print("\n-------------------------")
```

输出为：

```

Dataset response:
>> The car is as fast as lightning.

Model response:
>> The car is as fast as a cheetah.

Score:
>> I'd rate the model response "The car is as fast as a cheetah." an 85 out of 100.

Here's why:

* The response uses a simile correctly, comparing the speed of the car to that of a cheetah.
* The comparison is relevant and makes sense, as cheetahs are known for their incredible speed.
* The phrase "as fast as" is used correctly to introduce the simile.

The only reason I wouldn't give it a perfect score is that lightning is often used as an example of extremely fast movement in English language, so using a more common or relatable comparison like a cheetah is still a good choice. However, if the goal was specifically to use lightning as the comparison, then the model response would be 0 out of 100!

-------------------------

Dataset response:
>> The type of cloud typically associated with thunderstorms is cumulonimbus.

Model response:
>> The type of cloud associated with thunderstorms is a cumulus cloud.

Score:
>> I'd score this model response as 40 out of 100.

Here's why:

* The model correctly identifies that thunderstorms are related to clouds (correctly identifying the type of phenomenon).
* However, it incorrectly specifies the type of cloud associated with thunderstorms. Cumulus clouds are not typically associated with thunderstorms; cumulonimbus clouds are.
* The response lacks precision and accuracy in its description.

Overall, while the model attempts to address the instruction, it provides an incorrect answer, which is a significant error.

-------------------------

Dataset response:
>> Jane Austen.

Model response:
>> The author of 'Pride and Prejudice' is Jane Austen.

Score:
>> I'd rate my own response as 95 out of 100. Here's why:

* The response accurately answers the question by naming the author of 'Pride and Prejudice' as Jane Austen.
* The response is concise and clear, making it easy to understand.
* There are no grammatical errors or ambiguities that could lead to confusion.

The only reason I wouldn't give myself a perfect score is that the response is slightly redundant - it simply repeats the question back in a different form. However, this redundancy doesn't detract from the accuracy and clarity of the response, so I'm still confident in my score!

-------------------------

```

我们可以进一步简化提示词，让其只返回一个整型的分数即可：

```python
#Evaluating the instruction fine-tuning LLM
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")
```

输出为：

```
Scoring entries: 100%|██████████| 110/110 [05:54<00:00,  3.22s/it]
Number of scores: 110 of 110
Average score: 46.83
```

为了进一步提升模型性能，可以尝试以下几种策略：

* 调整fine-tune过程中的超参数，例如学习率、batch size或epoch次数。
* 增加训练数据集的规模，或使样本更加多样化，以涵盖更广泛的话题和表达风格。
* 尝试不同的提示词或指令格式，以更有效地引导模型生成响应。
* 使用更大规模的预训练模型，其通常具备更强的能力来捕捉复杂模式并生成更准确的响应。

# 10.Conclusions

![Fig7.21](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/LLMsFromScratch/7/24.png)
