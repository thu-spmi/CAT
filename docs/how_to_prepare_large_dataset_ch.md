[English](./how_to_prepare_large_dataset.md) | [中文](./how_to_prepare_large_dataset_ch.md)

# 大数据集数据的处理、准备和训练

## 背景

过去在小数据集上，我们处理音频数据的方式是：

1. 使用kaldi/torchaudio进行数据预处理，生成`.ark/.scp`文件，其中`.ark`文件存放特征的二进制文件，`.scp`文件为`.ark`文件的**索引**，此外还有一个`text`文件作为音频对应的标注；

2. 将数据打包为便于python读取的格式，具体代码可以参考 [code](../cat/utils/pipeline/asr.py#L20)。这一过程中，我们会保存特征帧长信息便于后续做动态batching；还会对label序列（通过tokenizer编码为数字）做padding，使其能够保存为`numpy.ndarray`格式；保存特征对应的索引（类似`.scp`文件）。最终保存的文件是上述几个文件的整合

这一过程中主要时间开销是在1中的特征处理阶段，2的时间开销非常小，处理1000小时的数据仅仅需要几分钟（受限于硬盘IO）。

在使用数据时（模型训练）时，数据加载基于`torch`标准的`Dataset`（map-style）接口开发，有以下几个特点：

1. 设置`shuffle=True`之后，数据加载是**完全随机**的：即任意两个句子都有可能出现在同一个mini-batch中；许多工作表明，mini-batch中句子的随机对NN模型性能是有利的

2. 尽管我们通过索引方式读取数据（而不是一次性把所有数据都从硬盘加载入内存），得益于OS级别的memory cache机制，一轮（一个epoch）迭代后，如果内存足够大，所有的数据都会在上层用户无感知的状态下被加载到内存中。后续的训练中事实上我们是直接从内存中读取数据，而不是硬盘，因此尽管特点1会导致大量随机读写，对内存上的读取而言，速度依然会非常快。

上述讨论中，我们事实上忽略了一种情况：如果memory不够大/数据太大呢？这个问题就引出了我们要讨论的普通数据加载方式的缺点：

如果内存不足以装下整个数据集，此时训练仍然能正常进行（得益于我们采用的索引方式加载，如果我们在训练开始前就将所有数据载入内存，那么会直接触发OOM导致的SIGKILL，使进程被杀死），但在OS层面出现了不同的情况：当系统内存被占满，而待读取的新数据不在内存中，OS会清理掉一些内存中的数据为新数据提供空间，而这一过程中数据加载是硬盘 --> 内存，是非常慢的（与内存 --> CPU/GPU相比）。由于NN模型训练往往需要迭代多轮，每一轮这个缓慢的数据加载都会导致时间的浪费。特别在`data size >> memory size`时，几乎等同于直接在硬盘上读取数据，而硬盘的随机读写性能更低（速度对比，sequential memory access >> random memory access > sequential disk access >> random disk acess），最终上层用户的感知就是，随着训练数据的增加，训练迭代的时间开销近乎指数级地增长，这在特别大数据集训练中是无法接受的。

## 方案

一个既定事实是，我们无法在硬件上无限地扩充去匹配更大的数据集（实际中大约1200小时的80-FBank数据就能占满256GB的内存），因此从硬盘中读取数据（而不是更快的内存中）是一个无法避免的问题。但是注意到，硬盘的顺序读取性能是远远大于随机读取性能的，我们可以从特点1出发，对数据加载加以改造。

[webdataset](https://github.com/webdataset/webdataset)提供的解决方案是：

减少数据加载的随机性，前面提到，完全顺序读取会对识别准确率会有一定的影响，但我们可以在二者之间取一个trade-off：将整个数据集划分为多个小文件（划分称为sharding，每个小文件即一个`.tar`文件），每个`.tar`文件中包含若干个句子（例如2000），在tar文件层级进行一次shuffle，在每个tar文件内再做一次 utterance级别shuffle，既保留一定的随机性，又能减少对硬盘的random access，可以显著提高IO性能。

基于`webdataset`，在处理大数据集（取决于内存大小，一般大于1500小时）时，我们将数据准备流程改造为：

1. 和普通方式1一致，特征的预处理；

2. 将特征和label（文本格式）每2000个句子打包为一个文件，进行处理。这一过程不涉及计算，主要是在做大量IO操作

当前为了兼容传统方式，1/2过程是分离的，后续可以考虑将1/2结合，进一步提高特征处理的效率。

**NOTE:**
和传统方式有比较大差异的是，label会被保存为文本格式，这是考虑到我们实际使用中可能会更换tokenizer，如果保存label的ID，就要把流程2再跑一次，这是非常不值得的。保存label的文本信息后，数据加载时直接由tokenizer做on-the-fly的编码，引入的额外开销也是基本可以忽略的。

特别要注意的是，使用某些tokenizer时，要处理好label中的空格，例如:

使用汉字建模的SentencePiece tokenizer（tokenizer训练时不带空格），如果这里数据准备时label中的空格没有去掉，就会被映射成`<unk>`，对模型性能造成严重影响，因此对中文数据集而言，最好先将label中空格去除，再进行数据sharding；
对一些空格不敏感的tokenizer（例如Jieba分词tokenizer），空格不会影响分词，因此没有关系。

后续可以考虑将音频特征和label分开独立处理，文本文件的处理比较轻松，一次性完整加载入内存也不会带来太多额外开销。

## 接口设计

### 数据准备
使用`webdataset`完成步骤2的代码可参考[code](../egs/wenetspeech/local/prep_wds.py#L16)。函数接口具体是

```python
# 每个文件保存的句子数，无特殊需要不用修改
UTTS_PER_FILE = 2000

def pack_data(
        # kaldi格式的.scp索引文件，支持多个文件以list形式传入
        f_scps: Union[List[str], str],
        # 文本文件，第一列为句子ID，和.scp文件中ID必须匹配，支持多个文件list传入
        f_labels: Union[List[str], str],
        # 输出文件夹
        d_out: str,
        # 输出文件格式
        fmt: str = "data-%05d.tar",
        # 长度分组配置，例如，"10:2000"表示仅保留长度在10-2000帧的句子，可以使用多个进行分组
        #           如，["10:500", "500:800", "800:1000", "1000:1200"]，不同长度组的文件会被保存在对应文件夹内
        filter_group: List[str] = None):
    ...
```

### 模型训练

在`hyper-p.json`文件中设置`train:option:large_dataset=True`使用，同时需要设定`train:option:tokenizer=xxx`和`train:option:trset=xxx`，`trset`为输出文件的格式匹配，例如：

```
# 在数据处理时，指定 d_out='./data', filter_group=['10:1000', '1000:2000']
# 则trset可以指定为：
# 1. 仅使用长度10-1000的句子
trset='./data/10_1000/data-*.tar'
# 2. 使用长度10-2000的句子
trset='./data/{10_1000,1000_2000}/data-*.tar'
# 3. 代码debug，仅使用10x2000个句子
trset='./data/10_1000/data-0000{0..9}.tar'
```

底层的`webdataset`接口调用在[code](../cat/shared/manager.py#L79)

**NOTE:** 由于开发集数据本身`shuffle=False`，且数据量一般较小，因此开发集数据仍然使用传统方式加载。

## DDP

上述所有讨论都建立在单机单卡的情况下，当涉及到DDP多卡或多机多卡训练时，这个问题会变得更复杂，例如：

```
trset="data-0{0..2}.tar"    # 包含3个tar文件，共3x2000句子
# 假设此时有两个进程（两块GPU）使用DDP训练
# 在tar文件层级做shuffle，把tar文件shuffle后分配到两个进程
gpu0: data-00.tar
gpu1: data-02.tar, data-01.tar
```

随之而来的问题是，两个进程上数据量不同，而DDP是同步式梯度更新训练，直接训练的话，gpu1会一直在等待gpu0同步，而gpu0已经完成所有数据遍历退出了。对这个问题，[wenet-e2e](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md#qa)提出的解决方案是使用`model.join()`。

我们使用更简单直接的方式：当一个进程遍历所有的数据后，直接强制所有进程结束当前迭代轮次（epoch），这样做使得1 epoch内训练的数据量减少了，但由于我们会迭代比较多轮次，并且每次会重新shuffle，这一部分带来的影响是比较小的。

> wenetspeech-L（～10000 hour）中包含约15 million句子，处理后得到约7500个`.tar`文件，使用8 GPU训练，7500 % 8 = 4，即每轮有4x2000句子被抛弃

**NOTE:**
上述例子只是为了便于理解，实际中webdataset会对tar文件做一些重复，使得tar文件层级能够被均分；但由于数据集句子无法被2000整除，会有一个（使用长度分组、多个数据集时会有多个）tar文件的句子相对其他较少，在最糟糕的情况下，会丢弃 `2000*(N-1)` 个句子，N 为总GPU数。

## 参考

1. [webdataset/webdataset: A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch. (github.com)](https://github.com/webdataset/webdataset)
2. [wenet/UIO.md at main · wenet-e2e/wenet (github.com)](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md)
3. [Distributed Training with Uneven Inputs Using the Join Context Manager — PyTorch Tutorials 1.10.1+cu102 documentation](https://pytorch.org/tutorials/advanced/generic_join.html#how-does-join-work)

