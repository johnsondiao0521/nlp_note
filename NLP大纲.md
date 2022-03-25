

## NLP大纲

### 顶会

1. 自然语言处理领域：**ACL、EMNLP、COLING、NAACL** 
2. 机器学习/深度学习领域：**ICML、NIPS、UAI、AISTATS、ICLR**
3. 数据挖掘领域：**KDD、WSDM、SDM**
4. 人工智能领域：**IJCAI、AAAI**
5. **Arxiv**

### 如何成为优秀的NLP人才

1. 扎实的数学基础、统计基础、数据结构与算法
2. 重视机器学习，理解核心的细节
3. 自然语言处理相关技术
4. 编程、编程、编程
5. 读论文/复现论文是AI工程师必备的能力

### NLP的四大分类

1. 序列标注：中文分词、词性标注、命名实体识别、语义角色标注...
2. 分类任务：文本分类、情感计算...
3. 句子关系判断：Entailment、QA、语义改写...
4. 生成式任务：机器翻译、文本摘要、写诗造句、看图说话...


在介绍Bert如何改造下游任务之前，先大致说下NLP的几类问题，说这个是为了强调Bert的普适性有多强。通常而言，绝大部分NLP问题可以归入上面的四类任务中：

**一类是序列标注**，这是最典型的NLP任务，比如中文分词、词性标注、命名实体识别、语义角色标注等都可以归入这一类问题，它的特点是句子中每个单词要求模型根据上下文都要给出一个分类类别。

**第二类是分类任务**，比如我们常见的文本分类、情感计算等都可以归入这一类。它的特点是不管文章有多长，总体给出一个分类类别即可。

**第三类任务是句子关系判断**，比如Entailment、QA、语义改写，自然语言推理等任务都是这个模式，它的特点是给定两个句子，模型判断出两个句子是否具备某种语义关系。

**第四类是生成式任务**，比如机器翻译、文本摘要、写诗造句、看图说话等都属于这一类。它的特点是输入文本内容后，需要自主生成另外一段文字。

![img](https://pic3.zhimg.com/80/v2-0245d07d9e227d1cb1091d96bf499032_720w.jpg)

对于种类如此繁多而且各具特点的下游NLP任务，Bert如何改造输入输出部分使得大部分NLP任务都可以使用Bert预训练好的模型参数呢？上图给出示例，对于句子关系类任务，很简单，和GPT类似，加上一个起始和终结符号，句子之间加个分隔符即可。对于输出来说，把第一个起始符号对应的Transformer最后一层位置上面串接一个softmax分类层即可。对于分类问题，与GPT一样，只需要增加起始和终结符号，输出部分和句子关系判断任务类似改造；对于序列标注问题，输入部分和单句分类是一样的，只需要输出部分Transformer最后一层每个单词对应位置都进行分类即可。从这里可以看出，上面列出的NLP四大任务里面，除了生成类任务外，Bert其它都覆盖到了，而且改造起来很简单直观。尽管Bert论文没有提，但是稍微动动脑子就可以想到，其实对于机器翻译或者文本摘要，聊天机器人这种生成式任务，同样可以稍作改造即可引入Bert的预训练成果。只需要附着在S2S结构上，encoder部分是个深度Transformer结构，decoder部分也是个深度Transformer结构。根据任务选择不同的预训练数据初始化encoder和decoder即可。这是相当直观的一种改造方法。当然，也可以更简单一点，比如直接在单个Transformer结构上加装隐层产生输出也是可以的。不论如何，从这里可以看出，NLP四大类任务都可以比较方便地改造成Bert能够接受的方式。这其实是Bert的非常大的优点，这意味着它几乎可以做任何NLP的下游任务，具备普适性，这是很强的。

##NLP预训练模型按照时间线有哪些

1. ELMo（2018.3 华盛顿大学）
   - 传统word2vec无法解决一次多义，语义信息不够丰富，诞生了**ELMo**
2. GPT（2018.06 OpenAI）
   - ELMo以LSTM堆积，串行且提取特征能力不够，诞生了**GPT**
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



[按时间线整理10种常见的预训练模型](https://zhuanlan.zhihu.com/p/210077100)



## NNLM

语言模型：

$$P(S) = P(w_1, w_2, ..., w_n)$$

$$P(w_1, w_2, ..., w_n) = P(w_1)P(w_2|w_1)P(w_3|w_1, w_2)...P(w_n|w_1, w_2,...w_{n-1})$$

$$L = \sum\limits_{w \in C}logP(w|context(w))$$

核心函数P的思想是根据句子里面前面的一系列前导单词预测后面跟哪个单词的概率大小，当预料中词典大小为100000，句子平均长度为5时，需要学习的参考大概100000*5-1个，为了降低计算复杂度，并考虑到词序列中离的更近的词通常在语义上也更相关，所以在计算时可以通过只使用n-1个词来近似计算，即n-grams：$$\hat{P}\left(W_1^T \right) = \prod_{t=1}^T\hat{P}\left(w_t \mid w_1^{t-1} \right)$$

n-grams存在的问题：

1. 泛化时常常有训练语料中没有出现过的词序列
2. 没有考虑词之间的相似性

NNLM：

1. 对词库里的每个词指定一个分布的词向量
2. 定义联合概率（国通序列中词对应的词向量）
3. 学习词向量和概率函数的参数



<u>众所周知，传统的模型预训练手段就是语言模型，比如ELMo模型就是以BiLSTM为基础架构、用两个方向的语言模型分别预训练两个方向的LSTM的，后面的OpenAI的GPT、GPT-2也是坚定不移地坚持着用祖传的（标准的、单向的）语言模型来预训练。</u>

然而，还有更多花样的预训练玩法。比如Bert就用了称之为“掩码语言模型（Masked Language Model）”的方式来预训练，

## Word2Vec



## Doc2Vec



## fastText

fastText涉及两个领域：一是使用fastText进行文本分类；二是使用fastText训练词向量

- fastText用作文本分类，做到了速度和精读的一个平衡：标准多核CPU情况下，不到十分钟，可以训练超过十亿个单词。不到一分钟，可以对50万个句子在312千个类别中进行分类。fastText的文本分类使用模型是CBOW的变种。

## ELMo

## Attention

**背景：**之前在做机器翻译Neural Machine Translation(NMT)模型中，通常的配置是Encoder-Decoder结构，encoder

$$\boldsymbol{X} \in \mathbb{R}^{n \times d}$$

Attention的数学形式为：

$$\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}\end{equation}$$

为了解决序列到序列模型记忆长序列能力不足的问题，一个非常直观的想法是，当要生成一个目标语言单词时，不光考虑前一个时刻的状态和已经生成的单词，还考虑当前要生成的单词和源语言句子的哪些单词更相关，即更关注源语言的哪些词，这种做法就是注意力机制。

## Self-Attention

- residual connection: 深度学习中经常使用，再探索一下深意
- Layer Norm: LN 做的事情比BN做的事情更简单点，不用考虑batch，输入一个向量输出另外一个向量，同一个example同一个feature不同dimension去计算mean和standard deviation
- Batch Norm: 同一个dimension不同example不同feature去计算mean和standard deviation
- NotImplementedError: Cannot convert a symbolic Tensor (strided_slice_6:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported

**Attention和Self-Attention的区别**

注意力机制的本质就是给予权重。Attention的权重是输入向量与输出向量之间的权重对应关系，Self-Attention的权重是输入向量内部之间的权重关系。

self-attention与普通attention的区别就是Q的不同，Q使用的是encoder的隐状态，而普通的attention使用的decoder的隐状态。常见的Cross Attention和Self Attention的区别就是Q的不同，但从最一般的角度来说，Q、K、V都可以有所不同。

其实Self Attention直接理解成引入了输入的两两交叉特征。

## 自回归(AR)和自编码(AE)

- **自回归**是时间序列分析或者信号处理领域喜欢用的一个术语，理解成语言模型就好了。一个句子的生成过程如下：首先根据概率分布生成第一个词，然后根据第一个词生成第二个词，然后根据前两个词生成第三个词，以此类推直到生成整个句子。
- **自编码**是一种无监督学习输入的特征方法：用一个神经网络输入变成一个低纬的特征，这是编码部分。然后再用一个Decoder尝试把特征恢复成原始的信号。例如，Bert看成一种AutoEncoder，它通过Mask改变了部分Token，然后试图通过其上下文的其他Token来恢复这些被Mask的Token。

## Transformer

[深入理解transformer源码](https://blog.csdn.net/zhaojc1995/article/details/109276945)

### Transformer结构

Transformer是一个升级版的seq2seq，也是由一个encoder和一个decoder组成的，encoder对输入序列进行编码，即x变成h，decoder对h进行编码，得到y，但encoder和decoder都不用RNN，二是换成了多个attention。

![](D:\vscode\bert-pics\transformer结构图.png)

![](D:\vscode\bert-pics\草稿上的transformer.png)

对上面的结构进行分开来看：

1. 左边的结构是Encoder、右边是Decoder
2. Encoder、Decoder的底部都是embedding，而embedding又分为两部分：input embedding和position embedding，Transformer抛弃了RNN，而RNN最大的优点就是在时间序列上对数据的抽象，所以文章中作者提出两种Positional Encoding的方法，将Encoding后的数据与embedding数据求和，加入了相对位置信息。
3. Encoder、Decoder的中部分是两个block，分别输入一个序列、输出一个序列，这两个block分别重复N次，Encoder的每个block里有两个子网，分别是multihead attention和feedforward network(ffn)；Decoder的block里有三个子网，分别是multihead attention和一个ffn。这些子网后面都跟了一个add&norm，即像resnet一样加一个。
4. decoder最后还有一个linear和softmax。

#### 1 Encoder

Encoder由6层相同的层组成，每一层由两部分组成；`multi-head、self-attention`和`position-wise、feed-forward network（是一个全连接层）`，两个部分都有一个残差链接(residual connection)，然后接着一个layer normalization。

#### 2 Decoder

Decoder也是由6个相同的层组成，每一个层包括三个部分：`multi-head,self-attention, mechanism`和`multi-head,context-attention,mechanism`和`position-wise,feed-forward network`，和Encoder一样，上面三个部分的每一个部分，都有一个残差连接，后接一个Layer Normalization。

Encoder和Decoder不同的地方在multi-head context-attention mechanism

#### 3 Attention

attention可以简单理解成encoder层的输出经过加权平均后再输入到decoder层中，它主要应用在seq2seq模型中，这个加权可以用矩阵来表示，也叫attention矩阵，它表示对于某一个时刻的输出y，它扎起输入y上各个部分的注意力，这个注意力就是刚才所说的加权，Attention 又分为很多种，其中两种比较典型的有加性 Attention 和乘性 Attention。加性 Attention 对于输入的隐状态 h_t 和输出的隐状态 s_t 直接做 concat 操作，得到 [s_t; h_t] ，乘性 Attention 则是对输入和输出做 dot 操作。

#### 4 Transformer评价

1. 并行计算，提高训练速度
   Transoformer用attention代替了原本的RNN，而RNN在训练的时候，当前的step的计算要依赖于上一个step的hidden state的，也就是说这是一个sequential procedure，即每次计算都要等之前的计算完成才能展开，而Transoformer不用RNN，所有的计算都可以并行计算，从而提高训练的速度

2. 建立直接的长距离依赖
   在Transoformer中，由于self attentionn的存在，任意两个word之间都有直接的交互，从而建立了直接的依赖，无论二者距离多远

   ​

### 20个QA

1. Transformer为何使用多头注意力机制？（为什么不使用一个头）

   注解：多头保证了Transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。[更详细的注解](https://www.zhihu.com/question/341222779)

2. Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘

   注解：使用Q/K/V不相同可以保证在不同空间进行投影，增强了表达能力，提高了泛化能力。[更详细的注解](https://www.zhihu.com/question/319339652)

3. Transformer计算attention的时候为何选择点乘而不是加法，两者计算复杂度和效果上有什么区别？

   注解：为了计算更快。矩阵加法在加法这一块的计算量确实简单，但是作为一个整体计算attention的时候相当于一个隐层，整体计算量和点积相似。在效果上来说，从实验分析，两者的效果和$d^k$相关，$d^k$越大，加法的效果越显著。

4. 为什么在进行**softmax**之前需要对attention进行scaled(为什么除以$d^k$的平方根)，并使用公式推导进行讲解

   注解：1.第三步和第四步是将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)；2.缩放的目的是作者认为过大的值会影响softmax函数，将其推入一个梯度很小的空间。  [更详细的注解](https://www.zhihu.com/question/339723385/answer/782509914)

5. 在计算attention score的时候如何对padding做mask操作？

   注解：padding位置置为负无穷(一般来说-1000就可以)。对于这一点，涉及到batch_size之类的。[更详细的注解]()

6. 为什么在进行多头注意力的时候需要对每个head进行降维？(可以参考上面一个问题)

7. 大概讲一下Transformer的Encoder模块？

8. 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？

9. 简单介绍一下Transformer的位置编码？有什么意义和优缺点？

10. 你还了解哪些关于位置编码的技术，各自的优缺点是什么？

11. 简单讲一下Transformer中的残差结构以及意义

12. 为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm在Transformer的位置是哪里？

13. 简单讲一下BatchNorm技术，以及它的优缺点。

14. 简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？

15. Encoder端和Decoder端是如何进行交互的？(在这里可以问一下seq2seq的attention知识)

16. Decoder阶段的多头自注意力和Encoder的多头自注意力有什么区别？(为什么需要decoder自注意力需要进行sequence mask)

17. Transformer的并行化体现在哪个地方？Decoder端可以做并行化吗？

18. 简单描述一下wordpiece model和byte pair encoding，有实际应用过吗？

19. Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout在测试的需要有什么需要注意的吗？

20. 引申一个关于bert问题，bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？

Tensor2Tensor

Transformer的编码器层有：Self-Attention、Add&Normal、Feed-Forward前馈层

像大部分NLP应用一样，首先将每个输入单词通过词嵌入算法转换为词向量，每个单词都被嵌入为512维的向量，一个编码器接受向量列表作为输入，接着将向量列表中的向量传递到自注意力层进行处理，然后传递到前馈神经网络层中，将输出结果传递到下一个编码器中。

词嵌入过程只发生在最底层的编码器中。所有的编码器都有一个相同的特点，即他们接受一个向量列表，列表中的每个向量大小为512维。在底层(最开始)编码中它就是词向量，但是在其他编码器中，它就是下一层编码器的输出(也是一个向量列表)。向量列表大小是我们可以设置的超参数----一般是训练集中最长句子的长度。

将输入序列进行词嵌入之后，每个单词都会流经编码器中的两个子层。接下来看Transformer的一个核心特性，在这里输入序列中每个位置的单词都有自己独特的路径流经编码器。在自注意力层中，这些路径之间存在依赖关系。而前馈(feed-forward)层没有这些依赖关系，因此在前馈层时可以并行执行各种路径。

Transformer的位置编码基于不同位置添加了正弦波，对于每个维度，波的频率和偏移都有不同。也就是说对于序列中不同位置的单词，对应不同的正余弦波，可以认为他们有相对关系。

```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)

        # 计算pe编码
        pe = torch.zeros(max_len, d_model) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]
```

一下总结：

NLP句子中长距离依赖特征的问题，self-attention天然就能解决这个问题，因为在集成信息的时候当前单词和句子中任意单词都能发生联系，所以一步到位就能把这个事情做掉了，不像RNN需要通过隐层节点序列往后传，也不像CNN需要通过增加网络深度来捕获远距离特征。



## Bert

### Bert在文本多分类任务的使用

Bert的全称：**Bidirectional Encoder Representations from Transformers**。现在对于Bert 的使用和衍生变种很多，比如：[Bert as service](https://github.com/hanxiao/bert-as-service)(可部署在服务器上，方便快捷版的Bert)、[Bert-NER](https://github.com/kyzhouhzau/BERT-NER)（用来做实体识别）、[Bert-utils](https://gitee.com/zengyy8/bert-utils)（句向量的引入）等等

假设你已经明确你有一个文本多分类的问题，第一步就是收集分类的手段，而不要考虑文本，这样做的好处一会再说。把问题拆解开来，分类手段有啥？决策树、朴素贝叶斯、SVM、高级点的LightGBM、神经网络等都可以，把它们记录下来，熟悉它们的使用方法（输入输出很重要），适应的范围和优缺点，原理的话，学有余力你可以学习。最快的方法是找代码直接看直接调通。

文本编码有啥方法呀？词袋模型，one-hot模型，TF-IDF方法，word2vec等，当然，你肯定还会看到Bert的身影，你把每种方法的实现、优缺点等搞懂，你就知道适合你的任务的编码方式是啥了。

你现在知道文本的编码有那么多、分类的方法有那么多，即便筛选掉那些你觉得肯定对你任务不合适的方法，这样排列组合的形式还是有很多。那么对任务上下游方法进行搭配就很重要了。谁也不会知道那个效果好，那就多搭配搭配试一试，比较各个模型组合起来的效果。对于分类问题比较好的衡量标准就是F-Score和准确率，一般来说准确率用的可能要多一些。但是在竞赛中F-Score是更为常见的。

（1）基于概率模型的朴素贝叶斯分类器：朴素贝叶斯分类器是一系列以假设特征之间强（朴素）独立下运用贝叶斯定理为基础的简单概率分类器。优点在于数据集较小的情况下的仍旧可以处理多类别问题使用于标称数据。面对Bert生成的序列向量，朴素贝叶斯并没有很好的处理能力，主要原因是：其一，Bert生成向量的各维度是连续属性；其二，Bert生成向量各个维度并不是完全独立的。因此这种分类方法在原理上来讲不会具有很好的分类效果；
（2）基于网络结构的SVM分类方法：支持向量机的思想来源于感知机，是一种简单的浅层神经网络。但SVM较一般的神经网络具有更好的解释性和更完美的数学理论支撑。SVM的目的在于寻求划分各类的超平面。当然，支持向量机可实现非线性分类，利用核函数将数据抽象到更高维的空间，进而将非线性问题转换为线性问题；
（3）基于树模型的LightGBM分类器：LightGBM是对一般决策树模型和梯度提升决策树模型XGBoost的一种改进。内部使用Histogram的决策树算法和带有深度限制的Leaf-wise的叶子生长策略，主要优势在于更快的训练效率、低内存占用以及适用大规模并行化数据处理。

- BERT和Transformer两者间是什么关系？

Transformer是特征抽取器，和CNN、RNN并列用于特征抽取的一种深层级网络结构，而BERT可视为一种两阶段的处理流程，这个流程使用的框架便是Transformer，再简单解释，你可以理解为BERT利用Transformer学会如何编码、存储信息知识。这是两者的关系。

在Transformer和BERT 之前，大家最常用的是CNN、RNN、Encoder-Decoder三大技术，覆盖了NLP领域80%的技术与应用，Transformer和BERT比它们好在哪里？每层网络学到了什么？多学了哪些知识？这些问题都是我一直在思考的，想在今天和大家分享一下目前的一些研究结论。



### 视频讲解目录

> 在b站上看的一个，顺便做了一些记录

1. Bert整体模型架构
2. 如何做预训练：MLM+NSP
3. 如何微调Bert，提升Bert在下游任务中的效果
4. 代码解析

#### 1. Bert整体模型架构

> Bert base是12层的Encoder层，Bert large是24层的Encoder层

Bert的position embedding使用的【0,1,2,3,4.....】而Transformer的position embedding使用的是正余弦函数

**Input = token embedding + segment embedding + position embedding**

**cls向量不能代表整个句子的语义信息**

- bert pretrain 模型直接拿来用作sentence embedding效果甚至不如word embedding，cls的embedding效果最差（也就是你说的pooled output）。把所有普通token embedding做pooling勉强能用（这个也是开源项目bert-as-service的默认做法），但也不会比word embedding更好。

#### 2. 如何做预训练：MLM+NSP

> MLM 详解：使用大量无标注的语料来训练

【对于无监督的目标函数来讲，有两种目标函数比较受重视】

1. 【MLM-掩码语言模型】autoregressive，自回归模型；只能考虑单侧的信息，典型的就是GPT
2. 【无监督目标函数】autoencoding，自编码模型：从损坏的输入数据中预测重建原始数据。可以使用上下文的信息，Bert就是使用的AE，其实和wordembedding的cbow模型很相似

【例子】

1. AR===>  P(我爱吃饭) = P(我)P(爱|我)P(吃|我爱)P(饭|我爱吃)
2. AE====> mask之后：【我爱mask饭】===> P(我爱吃饭|我爱mask饭) = P(mask=吃|我爱饭)

【本质是打破了文本，让他文本重建】

【mask模型的缺点】

1. 比如：**我爱吃饭**，**mask之后：【我爱mask mask】**
2. 优化目标：P(我爱吃饭|我爱mask mask) = P(吃|我爱)P(饭|我爱)

【mask概率问题】随机mask15% 的单词

1. 10% 替换为其他
2. 10%原封不动
3. 80%替换为mask

```
#mask代码实践
for index in mask_indices:
	if random.random() < 0.8:
		masked_token = "[MASK]"
	else:
		if random.random() < 0.5:
			masked_token = tokens[index]
		else:
			masked_token = random.choice(vocab_list)
```



------

> NSP任务详解

【NSP样本如下：】

1. 从训练语料库中取出两个连续的段落作为正样本
2. 从不同的文档中随机创建一对段落作为负样本

【**缺点：**主题预测和连贯性预测合并为一个单项任务】



#### 3. 如何微调Bert，提升Bert在下游任务中的效果

> **四种微调任务**
>
> 1. 句子对的分类任务（cls的输出，判断是否相似，0：不相似，1：相似）
> 2. 单个句子的分类任务 √ （cls的输出，做微调，二分类，多分类）
> 3. 问答集 √
> 4. 序列标注如NER √（解析：所有token输出，然后输入softmax，分类）

> 【说明】以那个单句分类为例， Bert finetuning的过程是先把之前pertrained model的参数导进去，然后输入句子，取第一个接softmax输出，然后lossfunction，之后梯度下降微调模型里的参数。

![](D:\vscode\Bert微调模型.png)

【如何提升Bert下游任务表现】

1. 获取谷歌中文Bert
2. 基于任务数据进行微调

【**做好训练的一些trick，相当于把上面的两个步骤扩展为下面例子的四个步骤**】

【例子：比如做微博文本情感分析】

1. 在大量通用预料上训练一个LM(pretrain)，example：中文谷歌Bert
2. 在相同领域上继续训练LM(Domain transfer)；在大量微博文本上继续训练这个Bert
3. 在任务相关的小数据上继续训练LM(Task transfer)，在微博情感文本上
4. 在任务相关数据上做具体任务(Fine-tune)

**总结：先Domain transfer再进行Task transfer最后Fine-tune性能是最好的**

【**pre-training的trick：如何在相同领域数据中进行further pre-training**】

1. 动态mask：就是每次epoch去训练的时候mask，而不是一直使用同一个
2. n-gram mask：其实比如ERNIE和SpanBert都是类似于做了实体词的mask

【**一些参数设置的trick**】

1. Batch size：16,32一般影响不太大
2. Learning rate(Adam)：5e-5,3e-5,2e-5,尽可能小一点避免灾难性遗忘
3. Number of epochs：3 or 4
4. Weight decay修改后的Adam，使用warmup，搭配线性衰减
5. 数据增强、自蒸馏、外部知识的融入

#### 4. 代码解析

[Pytorch微调Bert代码做文本分类](https://github.com/DA-southampton/Read_Bert_Code)



## BERT的通俗理解

### 1.预训练模型

​	BERT是一个预训练的模型，那么什么是预训练呢？举例子进行简单的介绍

​	假设已有A训练集，先用A对网络进行预训练，在A任务上学会网络参数，然后保存以备后用。当来一个新的任务B，采取相同的网络结构，网络参数初始化的时候可以加载A学习好的参数，其他的高层参数随机初始化，之后用B任务的训练数据来训练网络，当加载的参数保持不变时，称为"frozen"，当加载的参数随着B任务的训练进行不断的改变，称为"fine-tuning"，即更好地把参数进行调整使得更适合当前的B任务。

​	**优点：当任务B的训练数据较少时，很难很好的训练网络，但是获得了A训练的参数，会比仅仅使用B训练的参数更优。**

- 预训练模型的优势：
  1. 预训练模型从大规模语料中学习知识，对下游任务帮助很大
  2. 预训练提供了一种更好的参数初始化方式，使得在目标任务上泛化能力更好，收敛速度更快
  3. 预训练可以认为是一种正则化手段，可以防止模型在小数据集上过拟合。


- 预训练语言模型到目前分为两个阶段：
  1. 预训练word embeddings。这个阶段只训练词向量，而且是静态的，是一种feature-base方式。典型例子为word2vec，glove。利用词向量进行token embedding，然后送入模型中。模型设计百花齐放，但主要以LSTM为主。
  2. 预训练上下文编码器。这个阶段基于上下文动态学习embedding和encoding。典型例子为ELMO、GPT、BERT。
- 预训练任务
  - 目前大部分都是基于监督学习来构建的，又分为基于上下文学习和对比学习两类

#### Task #1：Masked LM

 为了训练双向特征，这里采用了Masked Language Model的预训练方法，随机mask句子中的部分token，然后训练模型来预测被去掉的token。具体操作是：**随机mask语料中15%的token，然后将masked token位置输出的final hidden vectors送入softmax，来预测masked token。**

这里也有一个小trick，如果都用标记[MASK]代替token会影响模型，所以在随机mask的时候采用以下策略：

1. 80%的单词用[MASK]token代替

   my dog is hairy → my dog is [MASK]

2. 10%单词用任意的词来进行代替

   my dog is hairy → my dog is apple

3. 10%单词不变

   my dog is hairy → my dog is hairy

#### Task #2：Next Sentence Prediction

为了让模型捕捉两个句子的联系，这里增加了Next Sentence Prediction的预训练方法，即给出两个句子A和B，B有一半的可能性是A的下一句，训练模型来预测B是不是A的下一句话

```
Input = [CLS] the man went to [MASK] store [SEP] penguin [MASK] are flight ## less birds [SEP]
Label = NotNext
             he bought a gallon [MASK] milk [SEP]
Label = IsNext
Input = [CLS] the man [MASK] to the store [SEP]
```

训练模型，使模型具备理解长秀霖上下文的联系的能力。



### 2. BERT模型

BERT：全称是**Bidirectional Encoder Representation from Transformers**，即双向Transformer的Encoder，BERT的模型架构基于多层双向转换解码，因为decoder是不能获要预测的信息的，模型的主要创新点都在pre-traing方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。其中“双向”表示模型在处理某一个词时，它能同时利用前面的词和后面的词两部分信息，这种“双向”的来源在于BERT与传统语言模型不同，它不是在给你大牛股所有前面词的条件下预测最可能的当前词，二是随机遮掩一些词，并利用所有没被遮掩的词进行预测。

下图展示了三种预训练模型，其中 BERT 和 ELMo 都使用双向信息，OpenAI GPT 使用单向信息

![](D:\vscode\bert-pics\三种预训练模型.png)

### 3. BERT的输入部分

![](D:\vscode\bert-pics\bert输入部分.png)

bert的输入部分是个线性序列，两个句子通过分隔符分割，最前面和最后增加两个标识符号。

**上图显示的是BERT输入表示，总述：输入嵌入分别是 token embeddings，segmentation embeddings和position embeddings的总和。**

BERT最主要的组成部分便是，词向量(token embeddings)、段向量(segment embeddings)、位置向量(position embeddings)

**每个单词有三个embedding:位置信息embedding，单词embedding，句子embedding**

- 位置信息embedding：这是因为NLP中单词顺序是很重要的特征，需要在这里对位置信息进行编码。
- 单词embedding：这个就是我们之前一直提到的单词embedding。
- 句子embedding：因为前面提到训练数据都是由两个句子构成的，那么每个句子有个句子整体的embedding项对应给每个单词。

把单词对应的三个embedding叠加，就形成了Bert的输入。

**注释：**如上图所示，输入有**A句[my dog is cute]**和**B句[he likes playing]**这两个自然句，我们首先需要将每个单词及特殊符号都转化为词嵌入向量，因为神经网络只能进行数值计算。其中特殊符[SEP]是用于分割两个句子的符号，前面半句会加上分割码A，后半句会加上分割码B。因为要建模句子之间的关系，BERT有一个任务是预测B句子是不是A句子后面的一句话，而这个分类任务会借助A/B句最前面的特殊符[CLS]实现，该特殊符可以视为汇集了整个输入序列的表征。**最后的位置编码是 Transformer 架构本身决定的**，因为基于完全注意力的方法并不能像 CNN 或 RNN 那样编码词与词之间的位置关系，但是正因为这种属性才能无视距离长短建模两个词之间的关系。因此为了令 Transformer 感知词与词之间的位置关系，我们需要使用位置编码给每个词加上位置信息。

**总结一下：**

1. token embeddings表示的是词向量，第一个单词是CLS，可以用于之后的分类任务
2. segment embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
3. position embedding表示位置信息

### 4.BERT的输出

想要获取bert模型的输出非常简单，使用`model.get_sequence_output()`和`model.get_pooled_output()`两个方法，但这两种翻翻针对NLP的任务需要进行一个选择

- 1、ouput_layer = model.get_sequence_output()，这个获取每个token的output输出[batch_size,seq_length, embedding_size]，如果做seq2seq或者ner用这个
- 2、output_layer = model.get_pooled_output()，这个输出是获取句子的output
- 3、注意：bert模型对输入的句子有一个最大长度，对于中文模型，我看到的是512个字。当我们用model.get_sequence_output()获取每个单词的词向量的时候注意，头尾[CLS]和[SEP]的向量。做NER和seq2seq的时候需要注意。

### 4.NLP的四大分类

1. 序列标注：中文分词、词性标注、命名实体识别、语义角色标注。。。
2. 分类任务：文本分类、情感计算。。。
3. 句子关系判断：Entailment、QA、语义改写。。。
4. 生成式任务：机器翻译、文本摘要、写诗造句、看图说话。。。

上图给出示例，对于句子关系类任务，和GPT类似，加上一个起始和终结符号，句子之间加个分隔符即可。

对于输出来说，把第一个起始符号对应的Transformer最后一层位置上面串接一个softmax分类层即可。

对于分类问题，与GPT一样，只需要增加起始和终结符号，输出部分和句子关系判断任务类似改造；

对于序列标注问题，输入部分和单句分类是一样的，只需要输出部分Transformer最后一层每个单词对应位置都进行分类即可。

从这里可以看出，上面列出的NLP四大任务里面，除了生成类任务外，Bert其它都覆盖到了，而且改造起来很简单直观。可观看这篇精彩文章  [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)

### 5.模型的评价

#### 优点

BERT是截止至2018年10月的最新的的state of the art模型，通过预训练和精调可以解决11项NLP的任务。使用的是Transformer，相对于rnn而言更加高效、能捕捉更长距离的依赖。与之前的预训练模型相比，它捕捉到的是真正意义上的bidirectional context信息

#### 缺点

作者在文中主要提到的就是MLM预训练时的mask问题：

1. [MASK]标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现;
2. 每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）





## RNN、LSTM、GRU

[RNN和LSTM](https://blog.csdn.net/zhaojc1995/article/details/80572098)

### Seq2Seq模型

#### 1. Seq2Seq模型简介

- Seq2Seq模型是RNN最重要的一个变种：N vs M（输入与输出序列长度不同），原始的N vs N要求序列等长，然后我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。

- Seq2Seq模型是输出的长度不确定时采用的模型，这种情况一般是在机器翻译的任务中出现，讲一句中文翻译成英文，那么这句英文的长度有可能会比中文短，也有可能比中文长，所以输出的长度就不确定了。如下图所示，输入的中文长度为4，输出的英文长度为2.

  ![](D:\vscode\seq2seq.png)

- 在网络结构中，输入一个中文序列，然后输出它对应的中文翻译，输出的部分的结果预测后面，根据上面的例子，也就是先输出“machine”，将"machine"作为下一次的输入，接着输出"learning",这样就能输出任意长的序列。机器翻译、人机对话、聊天机器人等等，这些都是应用在当今社会都或多或少的运用到了我们这里所说的Seq2Seq。

#### 2. Seq2Seq结构

- Seq2Seq属于Encoder-Decoder结构的一种，基本上就是利用两个RNN，一个RNN作为Encoder，另一个RNN作为Decoder。**Encoder负责将输入序列压缩成指定长度的向量，**这个向量就可以看成是这个序列的语义，这个过程称为编码，**获取语义向量最简单的方式就是直接将最后一个输入的隐状态作为语义向量c**。也可以对最后一个隐含状态做一个变换得到语义向量，还可以将输入序列的所有隐含状态做一个变换得到语义变量。

  ![语义向量只作初始化参数参与运算](D:\vscode\RNN网络.png)

- 而**decoder则负责根据语义向量生成指定的序列**，这个过程也称为解码，如下图，最简单的方式是将encoder得到的语义变量作为初始状态输入到decoder的RNN中，得到输出序列。可以看到上一时刻的输出会作为当前时刻的输入，而且其中语义向量C只作为初始状态参与运算，后面的运算都与语义向量C无关。

  ![](D:\vscode\语义向量只作初始化参数参与运算.png)

- decoder处理方式还有另外一种，就是语义向量C参与了序列所有时刻的运算，如下图，上一时刻的输出仍然作为当前时刻的输入，但语义向量C会参与所有时刻的运算。

  ![语义向量参与解码的每一个过程](D:\vscode\语义向量参与解码的每一个过程.png)

#### 3. 训练Seq2Seq模型

​

#### 4. Seq2Seq的应用

> 由于这种Encoder-Encoder结构不限制输入和输出的序列长度，因此应用的范围非常广泛，比如

- **机器翻译：**Encoder-Decoder的最经典应用，实际上这一结构就是在机器翻译领域最先提出的
- **文本摘要：**输入是一段文本序列，输出是这段文本序列的摘要序列。
- **阅读理解：**将输入的文章和问题分别编码，再对其进行编码得到问题的答案。
- **语音识别：**输入是语音信号序列，输出是文字序列。



**Automated Template Generation for Question Answering over Knowledge Graphs**



### 99.科学空间

分类模型本质上是在做**拟合**——模型其实就是一个函数（或者一簇函数），里边有一些待定的参数，根据已有的数据，确定损失函数（最常见的损失函数，就是误差平方和，不清楚的读者，可以回忆最小二乘法的过程。），然后优化损失函数到尽可能小，从而求出待定参数值。求出参数值之后，就可以用这个函数去进行一些预测。这便是分类的基本思想了，至于防止过拟合之类的，属于细节问题，在此先不作讨论。



神经网络解决的是第二个问题：这个函数是什么。传统的模型，如线性回归、逻辑回归，基本都是我们人工指定这个函数的形式，可是非线性函数那么多，简单地给定几个现成的函数，拟合效果往往有限，而且拟合效果很大程度上取决于找到了良好的特征——也就是还没有解决的第一个问题——函数的自变量是什么。（举个例子来说，一个函数如果是y=x2+xy=x2+x，是二次的非线性函数，那么如果用线性回归来拟合它，那么效果怎么也不会好的，可是，我定义一个新的特征（自变量）t=x2t=x2，那么y=t+xy=t+x是关于t,xt,x的线性函数，这时候用线性模型就可以解决它，问题是在不知道yy的具体形式下，怎么找到特征t=x2t=x2呢？这基本靠经验和运气了。）



为了解决“这个函数是什么”的问题，可以有多种想法，比如我们已经学过了泰勒级数，知道一般的非线性函数都能通过泰勒展式逼近。于是很自然的一个想法是：为什么不用高次多项式来逼近呢？高次多项式确实是个不错的想法，可是有学过计算方法的同学大概都知道，多项式拟合的问题是在训练数据内拟合效果很好，可是测试效果不好，也就是容易出现过拟合的现象。那么，还有没有其他办法呢？有！那位神经网络的发表者说——用复合函数来拟合！




## Sbert

1. sbert为什么速度会快？本质上两个句子不还是要通入到bert得到文本向量吗？理论上孪生bert应该比单bert更耗时才对。

   解答：首先这个快，其实是针对多个句子之间判断相似度而言的，比如我们有10个句子，需要判断这10个句子中两两句子之间的相似度，如果用传统的BERT，我们需要两两句子进行组合，拼接成一个长句子，输入到模型中，然后模型输出相似度。这样组合的话，排列组合C10/2的样本数可是远大于10个的。而用sbert，我们主需要向SBERT中输入10次句子，得的每个句子的embedding，然后利用embedding使用距离函数判断相似度即可。

   其次，为什么利用SBERT可以先直接通过模型得到每个句子的embedding然后再进行判断这种方式，而BERT不行的？BERT不是不可以采用这种方式，而是BERT用这种方式的效果并不好，这也是这篇文章提出的原因。因为BERT可能只是在大量的文本上进行了训练，但是并未专门针对语义相近的句子进行训练，所以即使两句语义很相近的句子，输入到BERT中，得到的两个句子的embedding之间的距离可能也会很大。

   孪生bert这个方式是这篇论文提出的一种训练BERT的方式，只是在训练的时候才会采用的，其实思想就是，利用这种方式，使得语义相近的两句句子，经过BERT模型后，输出的embedding在向量空间中近可能接近。经过训练后的SBERT，它还是一个单BERT，模型结构并没有什么改变。

2.  为什么bert语言模型会由于词频差异导致存在学到的embedding非凸/不连续，而CBOW不会呢？

   cbow学出来的embedding有没有词频偏差我不知道，但cbow有线性的性质，因为cbow的训练过程没有非线性激活，所以才有国王-女王=男性-女性的线性关系，BERT就没有这样的关系，所以cbow向量直接计算余弦相似度会比BERT更靠谱。

   cbow和skip-gram也有类似的问题，所以用词向量搞相似性计算、聚类时一般会先L2归一化到超球面上。

3. 为什么把句向量映射到高斯分布之后，向量之间的关系依然能保持呢？

   我认为并不能保持，如果变换前后向量的相对位置不变的话，指标就不会提升了，将处于错误位置的向量映射到正确的位置，准确度才会有改进啊，标准化流起到的就是修正的作用。

   相对位置不会完全不变，但是整体拓扑关系基本不变，或者说这个映射可以保证连续性，比如对2个原先的向量取插值，得到的新向量也基本是2个新向量的插值。这种关系不知道怎么保证的。

   标准化流是可微的，否则没法反向传播，所以标准化流是连续的双射，变换前后的空间肯定是同胚的。


---

## 知识图谱

**数据→信息→知识→洞见→智慧→影响力**

基本概念：本体、实体、属性、关系

区别点： 1.知识可推理；2.计算机看得懂。

三元组：

数据库：Neo4j

知识图谱的构建方法：

1. 本体定义
2. 知识抽取
3. 知识融合
4. 知识存储
5. 知识推理

知识图谱项目开发流程：

1. 知识建模
2. 知识获取
3. 知识融合
4. 知识存储
5. 知识计算
6. 知识应用



## [文章分类存档](https://github.com/panyang/AINLP-Archive)

[苏剑林博客](https://kexue.fm/category/Big-Data/31/)

[Bert源代码解读](https://github.com/DA-southampton/Read_Bert_Code)

[张俊林：BERT和Transformer到底学到了什么](https://baijiahao.baidu.com/s?id=1647838632229443719&wfr=spider&for=pc)

### 中文分词/词性标注

[结巴中文分词官方文档分析](https://www.cnblogs.com/baiboy/p/jieba1.html)

### 如何学习NLP和相关学习资源

[根据NLP历史学习](https://www.jianshu.com/u/abfe703a00fe)

[NLP个人技术实战心得](https://www.jianshu.com/u/abfe703a00fe)

[github-BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)

[Name-Entity-Recognition](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER)

https://github.com/panyang/AINLP-Archive

[RNN](https://blog.csdn.net/zhaojc1995/article/details/80572098)

[10中常见的预训练模型](https://zhuanlan.zhihu.com/p/210077100)

[NLP学习指南](https://github.com/leerumor/nlp_tutorial)



### Hugginface

[git hf](https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP)

[官网hf学习](https://huggingface.co/course/chapter1/3?fw=pt)

```
There are three main steps involved when you pass some text to a pipeline:

1. The text is preprocessed into a format the model can understand.
2. The preprocessed inputs are passed to the model.
3. The predictions of the model are post-processed, so you can make sense of them.
Some of the currently available pipelines are:

Some of the currently available pipelines are:
1. feature-extraction (get the vector representation of a text)
2. fill-mask
3. ner (named entity recognition)
4. question-answering
5. sentiment-analysis
6. summarization
7. text-generation
8. translation
9. zero-shot-classification
```



[Huggingface Transformer教程](https://fancyerii.github.io/2021/05/11/huggingface-transformers-1)

###Transformer

> 对transformer理解分3个层次：1.阅读原论文，2.图解transformer，3.阅读源码

[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)

[BERT大火却不懂Transformer？读这一篇就够了](https://zhuanlan.zhihu.com/p/54356280)

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

[The Illustrated Transformer【译】](https://blog.csdn.net/yujianmin1990/article/details/85221271)

[深入理解transformer源码](https://blog.csdn.net/zhaojc1995/article/details/109276945)

[举个例子讲下transformer的输入输出细节及其他](https://zhuanlan.zhihu.com/p/166608727)

https://huggingface.co/course/chapter1/3?fw=pt

###深度学习基础



### 词向量

[句子嵌入技术](https://zhuanlan.zhihu.com/p/268695012)

###预训练语言模型

BERT基础

[关于BERT，面试官们都怎么问](https://mp.weixin.qq.com/s/9ABJeU4skxTLPRNLLqUrIQ)

BERT源码

[Bert源码阅读](http://www.manongjc.com/article/30232.html)

###BERT应用

[BERT相关论文、文章和代码资源汇总](https://blog.csdn.net/yangfengling1023/article/details/85054871)



BERT实战



BERT之外



###命名实体识别

[实体识别NER——BiLSTM+CRF知识总结与代码（Pytorch）分析——细粒度实体的识别(基于CLUENER)](https://blog.csdn.net/weixin_43038752/article/details/112257793)

###文本分类/情感分析



###文本摘要



###主题挖掘

[如何用简单易懂的例子解释条件随机场（CRF）模型？它和HMM有什么区别？](https://www.zhihu.com/question/35866596/answer/236886066)

[如何轻松愉快地理解条件随机场（CRF）](https://zhuanlan.zhihu.com/p/104562658)



###文本匹配



###问答系统/对话系统/聊天机器人



###文本生成



###阅读理解



### 搭建搜索引擎

https://bitjoy.net/2016/01/04/introduction-to-building-a-search-engine-1/



从wordembedding到Bert模型的自然语言处理的预训练技术发展史

1. Word Embedding考古史：语言模型 $$L = $$
2. ​



