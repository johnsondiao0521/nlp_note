

### 一、NLP预训练模型按照时间线有哪些？

1. ELMO（2018.3 华盛顿大学）
   - 传统word2vec无法解决一次多义，语义信息不够丰富，诞生了**ELMO**
2. GPT（2018.06 OpenAI）
   - ELMO以LSTM堆积，串行且提取特征能力不够，诞生了**GPT**
3. BERT（2018.10 Google）
   - GPT虽然用transformer堆积，但是是单向的，诞生了**BERT**
4. XLNet（2019.06 CMU+Google brain）
   - BERT虽然双向，但是mask不适用于自编码模型，诞生了**XLNET**
5. ERNIE（2019.4）
   - BERT中mask代替单个字符而非实体或短语，没有考虑词法结构/语法结构，诞生了**ERNIE**
6. BERT-wwm（2019.6.30 哈工大+讯飞）
   - 为了mask掉中文的词而非字，让BERT更好的而应用在中文任务，诞生了**BERT-wwm**
7. RoBERTa（2019.7.26 Facebook）
   - Bert训练用更多的数据、训练步数、更大的批次，mask机制变为动态的，诞生了**RoBERTa**
8. ERNIE2.0（2019.7.29 百度）
   - ERNIE的基础上，用大量数据和先验知识，进行多任务的持续学习，诞生了**ERNIE2.0**
9. BERT-wwm-ext（2019.7.30 哈工大+讯飞）
   - BERT-wwm增加了训练数据集、训练步数，诞生了**BERT-wwm-ext**
10. ALBERT（2019.10 Google）
    - BERT的其他改进模型基本靠增加参数和训练数据，考虑轻量化之后，诞生了**ALBERT**

### 二、transformers、pytorch-transformers、pytorch-pretrained-bert三者的关系

`transformers`包也叫`pytorch-transformers`或者`pytorch-pretrained-bert`，实际上transformers库是最新的版本（以前称为pytorch-transformers和pytorch-pretrained-bert），所以它在前两者的基础上对一些函数与方法进行了改进，包括一些函数可能只有在transformers库里才能使用，因此使用transformers库比较方便。它提供了一些列的STOA模型的实现，包括(Bert、XLNet、RoBERTa等)。

### 三、安装transformer库：
**pip install transformer**

### 四、Bert的模型有哪些？在哪里？
- 在使用transformers的时候，由于Bert的文件都在AWS上存储，transformers的默认下载地址指向的是AWS，因此在国内下载速度非常慢。需要手动下载。
  
模型的bin文件下载
```
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}

```

模型的json文件下载

```
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-config.json",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-config.json",
}
```

模型的词表文件下载:

```
PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        'bert-base-german-cased': "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
        'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
        'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
    }
}
```

`transformers的模型下载默认在~/.cache/torch/transformers/中！`

### 未命名的记录

- 预训练模型的优势：
  1. 预训练模型从大规模语料中学习知识，对下游任务帮助很大
  2. 预训练提供了一种更好的参数初始化方式，使得在目标任务上泛化能力更好，收敛速度更快
  3. 预训练可以认为是一种正则化手段，可以防止模型在小数据集上过拟合。


- 预训练语言模型到目前分为两个阶段：
  1. 预训练word embeddings。这个阶段只训练词向量，而且是静态的，是一种feature-base方式。典型例子为word2vec，glove。利用词向量进行token embedding，然后送入模型中。模型设计百花齐放，但主要以LSTM为主。
  2. 预训练上下文编码器。这个阶段基于上下文动态学习embedding和encoding。典型例子为ELMO、GPT、BERT。
- 预训练任务
  - 目前大部分都是基于监督学习来构建的，又分为基于上下文学习和对比学习两类




