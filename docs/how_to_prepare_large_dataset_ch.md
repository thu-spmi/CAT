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

上述 1 & 2 步骤可以分别进行，也可以同时完成，具体可参考使用示例。

**NOTE:**
和传统方式有比较大差异的是，label会被保存为文本格式。在模型训练中，数据加载时直接由tokenizer做on-the-fly的编码，引入的额外开销也是基本可以忽略的。

特别要注意的是，使用某些tokenizer时，要处理好label中的空格，例如:

使用汉字建模的SentencePiece tokenizer（tokenizer训练时不带空格），如果这里数据准备时label中的空格没有去掉，就会被映射成`<unk>`，对模型性能造成严重影响，因此对中文数据集而言，最好先将label中空格去除，再进行数据sharding；
对一些空格不敏感的tokenizer（例如Jieba分词tokenizer），空格不会影响分词，因此没有关系。

## 使用示例

参考实验[yesno](../egs/TEMPLATE/exp/asr-ctc-large-corpora)

**NOTE:** 

- 由于开发集数据本身`shuffle=False`，且数据量一般较小，因此开发集数据仍然使用传统方式加载；
- 在大规模数据训练中，**epoch** 的概念不再存在，数据是以数据流的形式不断传入 dataloader；因此在训练中，日志输出的 epoch id 总是 1，我们无法严格准确地获取当前 epoch 数目，但可以通过以下方式估算
   
   ```
   num_epochs = num_steps * batch_size / num_total_utts
   ```

- 在大规模数据训练中，从训练中断点恢复（`--resume`）是不严格的恢复，会难以避免地导致一部分数据被模型学习更多次（如果不是频繁中断+恢复，影响理应不大）。


## 参考

1. [webdataset/webdataset: A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch. (github.com)](https://github.com/webdataset/webdataset)
2. [wenet/UIO.md at main · wenet-e2e/wenet (github.com)](https://github.com/wenet-e2e/wenet/blob/main/docs/UIO.md)
3. [Distributed Training with Uneven Inputs Using the Join Context Manager — PyTorch Tutorials 1.10.1+cu102 documentation](https://pytorch.org/tutorials/advanced/generic_join.html#how-does-join-work)

