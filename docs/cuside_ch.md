[English](./cuside.md) | [中文](./cuside_ch.md)

### 流式语音识别

流式语音识别，指的是在说话人讲话的同时进行识别，而不是等到说话人讲完整句话后再开始识别。

然而，目前业界常用的神经网络结构，例如基于自注意力机制的transformer和conformer，均依赖整句作为输入，因此不适用于低延迟语音识别。

为了解决这一问题，很多系统采用了分块（chunk）的模型。具体而言，一句话会被切分为多个块，然后再送入神经网络逐块进行识别，这样就将延迟降低为一个块的长度。

### 上下文感知块

在基于块的低延迟语音识别模型中，一个常见做法是为每个块附加一定的历史帧和未来帧，以提供上下文信息，构成上下文感知块(context sensitive chunk)。

已有的工作表明，上下文信息对精确的声学建模至关重要，上下文信息的缺失将造成10%以上的识别准确率损失。

但是，为了获取未来信息，模型必须等到一定数量的未来帧到达后再开始识别，这显著增加了识别延迟。

为了解决这一问题，我们提出了一种基于分块、预测未来、解码（Chunking, Simulating future context and Decoding，CUSIDE）的低延迟语音识别框架。

### CUSIDE

在CUSIDE模型中，模型使用模拟的未来帧而不是真实未来帧进行识别，由此可以免除对未来信息的依赖，减小识别延迟。

模拟帧使用一个生成器以流式的方式生成，该生成器由生成编码器和生成预测器构成，可以以无监督方式进行训练(因为将输入帧向前移动即可得到对应的预测目标，这里我们受到了无监督表征学习方法APC的启发)，不需要额外的标注信息。

此外，我们还通过流式、非流式模型共享参数和联合训练等方法(unified streaming/non-streaming model)，减小了流式模型和非流式模型之间的性能差距，在aishell1数据集达到了业界领先的结果。

### 代码说明

核心代码位于`cat/rnnt/train_unified.py`，现对其参数进行简要说明：
- `downsampling_ratio`：输入encoder前后的降采样率，默认为8。
- `chunk_size`：默认的chunk大小（帧），默认值为40。
- `context_size_left`：默认的left context帧数，默认值40。
- `context_size_right`：right context帧数，默认值40。
- `jitter_range`：训练时我们对chunk的大小进行了抖动(chunk size jitter)，jitter_range决定了抖动的范围（注意是chunk经过encoder输出后的抖动范围，对应输入前的抖动范围是jitter_range*downsampling_ratio）。默认值为2。        
- `mel_dim`：梅尔谱维度，默认为80。
- `simu`：是否使用模拟未来帧。默认的生成编码器是单向GRU，默认的生成预测器是一个简单的前馈神经网络。
- `simu_loss_weight`：对合成损失的加权系数。合成损失默认是一个L1 loss。

对训练流程的说明：

- 在训练过程中，流式模型和非流式模型进行参数共享和联合训练。对于每一条样本，会分别按流式方式和非流式方式进行识别并计算对应的loss。

- `chunk_forward()`函数是训练时按流式进行forward的函数，`chunk_infer()`函数是测试时按流式进行forward的函数。两者的不同在于，训练时使用的chunk size jitter和使用随机的未来信息（未来信息随机从a.模拟的未来帧；b.不使用未来帧；c.真实的未来帧三种情况中选择），而测试时使用固定的chunk size，不使用随机未来信息，而是根据设置固定选择三种情况之一。

- `chunk_forward()`和`chunk_infer()`中均涉及的关键操作是划分chunk以及为每个chunk拼接上下文。划分chunk通过将原句子reshape实现，上下文则通过上下文和当前chunk之间的关系，通过左移或右移chunk序列得到：例如考虑left context, chunk size, right context均为40的情况，那么当前chunk的left contxt实际上就是上一chunk,而当前chunk的right context实际上就是下一chunk。如果使用模拟的未来帧，则当前chunk的right context为simu net接收当前chunk后的输出。

- 训练时我们一次性模拟得到所有chunk的future context，而不是逐块进行模拟。

- `forward()`函数中也包括了按非流式方式进行forward的计算。其计算方式与训练一个标准的非流式模型相同，此处不再赘述。



