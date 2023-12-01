# 多通道端到端流式语音识别

**本文档介绍如何使用多通道端到端语音识别 (ME2E ASR) 框架进行多通道端到端流式语音识别任务**

*相关文献：*

>*1.Ochiai, Tsubasa, et al. "Multichannel end-to-end speech recognition." International conference on machine learning. PMLR, 2017.<br>
2.Keyu An, Hongyu Xiang, Zhijian Ou:
"CAT: A CTC-CRF Based ASR Toolkit Bridging the Hybrid and the End-to-End Approaches Towards Data Efficiency and Low Latency." INTERSPEECH 2020: 566-570 <br>
3.Keyu An, Huahuan Zheng, Zhijian Ou, Hongyu Xiang, Ke Ding, Guanglu Wan."CUSIDE: Chunking, Simulating Future Context and Decoding for Streaming ASR."INTERSPEECH, 2022*

实验参考了单通道端到端语音识别中的egs/aishell/ctc-v1实验，前端网络程序位于文件夹cat/front下

**本文档将从以下步骤说明前期数据准备与ME2E 流式ASR相关的参数配置**

+ 多通道端到端流式语音识别
  + 框架讲解
  + 代码说明
  + 数据准备
  + 训练与测试

## 框架讲解

多通道端到端语音识别框架由三部分组成：<br>
- 多通道语音信号处理 -> 特征转换 -> 语音识别<br>

下面我们分别介绍：

**1、多通道语音信号处理**：

本实验中使用基于mask的波束形成方法进行多通道语音信号处理，该方法使用BLSTM网络来估计单通道的语音和噪声掩码，接着在通道轴上进行平均，得到单通道的掩码，用单通道的掩码计算语音和噪声的空间协方差矩阵，进而估计波束形成滤波器系数，最后进行波束形成，得到增强后的单通道语音，其流程图如下所示：

  <div align="center">
    <img src="MVDR.png" alt="MVDR Beamforming">
  </div>

  + T-F掩码估计：

    通过两个神经网络估计信号和噪声的掩码，其中Xc是(B,T,F,C)的矩阵，分别代表(batch size, time, frequency, channels)

    $$\mathbf{m}_{\text{S}}(c) = \text{BLSTM}\left( \lvert \mathbf{X}(c) \rvert \right) $$

    $$\mathbf{m}_{\text{N}}(c) = \text{BLSTM}\left( \lvert \mathbf{X}(c) \rvert \right) $$

  + 跨通道功率谱密度估计:

    其中 $\mathbf{x}(t, f)=\{x(t, f, c)\}_{c=1}^C \in \mathbb{C}^C, \mathrm{~T}$ 是输入特征的长度, $m_S(t, f) \in[0,1]$ 是语音信号的时频掩模 (time-frequency mask) , $m_N(t, f) \in[0,1]$ 是噪声信号的时频掩模, 通过如下方式得到，其中 $m_S(t, f)$ 和 $m_N(t, f)$ 是分别估计得到的, 两者的加和并不一定等于1:

  <div align="center">
    <a href="https://latex.codecogs.com/svg.latex?\begin{aligned}\boldsymbol{\Phi}_{\mathrm{Ss}}(f) & =\frac{1}{\sum_{t=1}^T m_S(t, f)} \sum_{t=1}^T m_S(t, f) \mathbf{x}(t, f) \mathbf{x}^{\dagger}(t, f) \\\boldsymbol{\Phi}_{\mathrm{NN}}(f) & =\frac{1}{\sum_{t=1}^T m_N(t, f)} \sum_{t=1}^T m_N(t, f) \mathbf{x}(t, f) \mathbf{x}^{\dagger}(t, f)\end{aligned}">
    <img src="PSD.png" alt="语音与噪声跨通道功率谱密度" width="400">
    </a>
  </div>

  $$ m_S(t, f) = \frac{1}{C} \sum_{c=1}^C m_S(t, f, c) $$

  $$ m_N(t, f) = \frac{1}{C} \sum_{c=1}^C m_N(t, f, c) $$
   

  + MVDR波束成形

    MVDR 通过对多阵列接收信号应用线性滤波器来降低噪声干扰并恢复信号分量:
    $$\hat{x}(t, f)=\sum_{c=1}^c h(f, c) \times x(t, f, c)$$
    其中 $x(t, f, c)$ 是第 $c$ 个麦克风接收信号的短时傅立叶变换在时刻 $t$ 和频点 $f$ 处的取值, $\hat{x}(t, f)$ 是增强后的信号在时刻 $t$ 和频点 $f$ 处的取值, $x(t, f, c)$ 和 $\hat{x}(t, f)$ 的取值均为复数, $\mathrm{C}$ 是麦克风个数。 $h(f, c)$ 是对应第 $\mathrm{c}$ 个麦克风的时不变（time invariant）滤波器系数在频点 $f$ 处的取值。MVDR 通过求解最小方差无失真响应(minimum variance distortionless response) 得到波束成形滤波器系数 $\mathbf{h}(f)=\{h(f, c)\}_{c=1}^C \in \mathbb{C}^C$ 其中:

    $\boldsymbol{\Phi}_{\mathrm{SS}}(f)$ 是信号的跨通道功率谱密度矩阵,

    $\boldsymbol{\Phi}_{\mathrm{NN}}(f)$ 是噪声的跨通道功率谱密度矩阵,

    $\mathbf{u}$ 是参考麦克风的 one-hot 向量，本方法默认选取第一个麦克风作为参考麦克风，
    
    $\mathrm{tr}$ 是矩阵求迹的操作。
  <div align="center">
    <a href="https://latex.codecogs.com/svg.latex?\mathbf{h}(f)=\frac{\boldsymbol{\Phi}_{\mathrm{NN}}^{-1}(f) \boldsymbol{\Phi}_{\mathrm{SS}}(f)}{\operatorname{tr}\left\{\boldsymbol{\Phi}_{\mathrm{NN}}^{-1}(f) \boldsymbol{\Phi}_{\mathrm{SS}}(f)\right\}} \mathbf{u}">
    <img src="h_f.png" alt="波束成形滤波器系数估计，GitHub无法显示，请点击查看" width="300">
    </a>
  </div>



**2、特征转换**：将语谱图特征转换为Fbank特征

**3、语音识别**：使用Comformer网络进行语音识别，此处不再赘述，有关流式语音识别的相关知识请参考[cuside_ch.md](https://github.com/thu-spmi/CAT/blob/master/docs/cuside_ch.md)



## 代码说明

核心代码位于```cat/ctc/train_me2e.py```与```cat/ctc/train_me2e_chunk.py```，关于流式的参数设置可参考[cuside_ch.md](https://github.com/thu-spmi/CAT/blob/master/docs/cuside_ch.md)，现主要介绍下有关前端和特征转换的参数配置：

+ n_fft：进行STFT变换时的点数, 默认为512
+ win_length：进行STFT变换时的窗长, 默认为None, 窗类型为hann
+ hop_length：进行STFT变换时的跳长, 默认128
+ idim：梅尔滤波器组的数量, 默认80
+ beamforming：使用的滤波器类型, 可以选择"mpdr"、"mvdr"、"gev"、"wpd", 默认为"mvdr"
+ wpe：是否使用去混响, 默认为False
+ ref_ch：进行波束形成时的参考信道，默认使用第一个通道0，当为"-1"时，使用注意力机制去选择参考信道

```cat/ctc/train_me2e.py```主要实现的是前端与后端进行联合训练的框架，输入的多通道信号首先经过self.beamforming(), 来进行多通道的增强，得到单通道的语音，计算fbank特征并返回；接着送入到后端ASR网络self.encoder()中进行识别，并计算得到整句的loss

```cat/ctc/train_me2e_chunk.py```主要实现的是流式的端到端语音识别框架，对于每一条样本，会分别按流式方式和非流式方式进行识别并计算对应的loss，框架结构如下所示：

  <div align="center">
    <img src="ME2E.png" alt="ME2E" width="500" >
  </div>

多通道的语音波形首先进行STFT变换，得到语谱图特征，进行分块处理后，送入到波束形成网络，得到单通道的分块语谱图，接着将单通道的分块语谱图拼接，得到单通道的整句语谱图，分别计算Fbank特征后，得到整句的loss，在计算分块loss前，可以进行模拟未来操作，然后送入到ASR网络，计算得到分块loss，
最终将分块loss与整句loss，以及simu loss(如果有)进行加和，得到总的loss

多通道语音增强的相关代码位于```cat/front/```下，下面进行简要的说明：

+ stft.py：STFT变换模块。pytorch中，torch.stft()函数无法直接对多通道的语音进行STFT变换，在此模块中，将输入(Batch, Nsample, Channels)变为(Batch * Channels, Nsample)，然后进行STFT变换，得到(Batch * Channel, Frames, Freq, 2=real_imag)，然后将其转换为 -> (Batch, Frame, Channel, Freq, 2=real_imag)

+ nets_utils.py：网络相关的一些功能性函数，如to_device()、pad_list()等等

+ multi2mono.py：若不使用增强网络，则使用该程序，选择多通道中的指定某通道，作为单通道的波形输出, 或可直接使用model.wave2fbank()，来获取默认第一个通道的fbank特征

+ log_mel.py：用于计算单通道的fbank特征，输入为单通道的语谱图特征(B, T, F)，输出为(B, T, D)，其中D指的是Mel频段的维度

+ beamformer_net.py：波束形成网络配置入口，可以选择是否使用去混响滤波器wpe，可以选择使用神经网络的方式：1、直接估计波束形成滤波器系数；2、估计各通道的掩码，在通道轴上进行平均得到滤波器系数，进而估计滤波器系数，根据约束条件不同，可以选择波束形成滤波器的类型有：mpdr、mvdr、gev、wpd

+ dnn_wpe_new.py：WPE滤波器的实现

+ dnn_beamformer.py：基于掩码的mpdr、mvdr、gev波束形成滤波器的实现

+ conv_beamformer.py：基于掩码的wpd波束形成滤波器的实现

+ filter_net.py：直接估计波束形成滤波器系数方法的实现

+ mask_estimator.py：基于掩码的方法中，掩码的估计网络，对每个通道计算其掩码

## 数据准备

本文档中实验选择开源的Aishell4数据作为实验数据，这些开源数据可以直接下载得到。下载好的数据由音频及训练、验证、测试文本构成。将train_S、train_M、train_L合成为一个train文件夹。
准备完成后，执行命令：

```bash
bash local/data_multi.sh -subsets train dev test -datapath /path/to/aishell4 
bash local/audio2ark_multi.sh train dev test --res 16000
```

其中：

+ data_multi.sh：用于提取Aishell4中无重叠说话人的片段，放在Aishell4目录下，进行数据集划分，将其划分为train dev test，dev与train的比例为1：20，最后生成wav.scp与text文件，放在data/src对应的文件夹中
+ audio2ark_multi.sh：用于将波形文件转换为特征文件，并生成.ark文件

更多的细节可以通过执行：

```bash
bash local/data_multi.sh -h
bash local/audio2ark_multi.sh -h
```

来获取其使用方法

### 模拟数据生成

如果多通道数据不足的话，还提供了```egs/aishell4/local/mix_gen.py```来进行多通道数据生成

该程序参考了ConferencingSpeech2021比赛中的生成设置，使用pyroomacoustics，采用图像法模拟RIR。房间大小从3∗3∗3 m3到8∗8∗3 m3。麦克风阵列随机放置在高度为1.0 ~ 1.5 m的房间内。声源包括语音和噪声，声源来自房间内的任何位置，高度范围为1.2 ~ 1.9 m。两个声源之间的夹角大于20°。声源与麦克风阵列之间的距离为0.5 ~ 5.0 m。目前可以选择的麦克风阵列类型有线性麦克风阵列与圆形麦克风阵列，麦克风数量为8。信噪比随机范围[-5, 10 )

使用方法如下：

运行前可以对程序```egs/aishell4/local/mix_gen.py```中的配置进行修改：

+ mic_type：麦克风类型，可选择circular 或 linear
+ channels：麦克风通道数
+ noise_num：噪声数量
+ num_room：房间数
+ utt_per_room：同一个房间最多生成几条混合语音
+ snr_range：SNR范围

使用指令运行：

```bash
python egs/aishell4/local/mix_gen.py /
  --speech_dir /Path/to/the/speech/directory /
  --noise_dir /Path/to/the/noise/directory /
  --output_dir /Path/to/the/output/directory
```

运行成功后，会在output_dir生成混合的.wav文件，命名为uid_speech-name_noise-names_room-size_snr_mic_type.wav

除此之外还会生成wav.scp与all_samples_info.json文件，其中all_samples_info.json内容如下：

```json
[
    {
        "idx": 0, //混合语音的id
        "simu_file": "0_F031-001_noise3_33.61:36.19_noise1_101.28:103.86_5.74_6.58_3.00_7.666_circular.wav",  //混合语音的文件名
        "all_speech_names": "F031-001", //干净语音的文件名
        "noise_names": [
            "noise3_33.61:36.19", //噪声的文件名与段落位置
            "noise1_101.28:103.86"
        ],
        "room_size": { // 房间尺寸
            "x": 5.744067519636624,
            "y": 6.5759468318620975,
            "z": 3
        },
        "snr": 7.6663277728757215, // 信噪比
        "mic_type": "circular", // 麦克风类型
        "speech_source": [ // 语音源位置
            3.462313530519645,
            3.5831228409633793,
            1.4965583595372332
        ],
        "noise_sources": [ // 噪声源位置
            [
                3.7100593959906845,
                2.877550235566273,
                1.824241100547456
            ],
            [
                5.535343962477329,
                2.521491040926764,
                1.7542075266578652
            ]
        ],
        "noise_interal": [ // 噪声取样的起始点与结束点
            "537813:579114",
            "1620409:1661710"
        ],
        "channels": 8 // 麦克风通道数量
    },

.....
]

```

## 训练与测试

参考cuside的训练方法，流式模型和非流式模型进行参数共享和联合训练。该代码位于```cat/ctc/train_me2e_chunk.py```下

配置好config.json与hyper-p.json文件后，使用指令(以/ctc-e2e-chunk为例)
```bash 
python utils/pipeline/asr exp/ctc-e2e-chunk --sta 1 --sto 3
```
来进行训练。

输入指令前请确保当前目录位于./CAT-ME2E/egs/aishell4, 并有足够的储存空间与计算资源

可以通过指令：
```bash
tensorboard --logdir=exp/ctc-e2e-chunk/log --port 6006
```
来查看训练进度

在进行测试时：

使用指令(以/ctc-e2e-chunk为例)
```bash 
python utils/pipeline/asr exp/ctc-e2e-chunk --sta 4 
```
来进行测试

+ 非流式过程依次使用```model.beamforming()```与```model.encoder()```来进行识别。其中model.beamforming()的作用是进行多通道的增强，得到单通道的语音，计算fbank特征并返回；model.encoder()的作用是将
+ 流式过程使用```model.bf_chunk_infer()```来进行流式的识别，具体的框架结构与流程参考上文[代码说明](#代码说明)

流式的训练和测试的区别与cuside中相同，训练时使用的chunk size jitter和使用随机的未来信息（未来信息随机从a.模拟的未来帧；b.不使用未来帧；c.真实的未来帧三种情况中选择），而测试时使用固定的chunk size，不使用随机未来信息，而是根据设置固定选择三种情况之一

关于chunk的划分方法，举例如下：

假设是一个二维张量，形状为 (batch*N_chunks, chunk_size)，其中batch =3, N_chunks = 2, chunk_size = 2, 设置left_context_size = 2, right_context_size = 2

Original Samples:
+ tensor([[ 1,  2],
      [ 3,  4],
      [ 5,  6],
      [ 7,  8],
      [ 9, 10],
      [11, 12]])

Left Context:
+ tensor([[ 0.,  0.],
        [ 1.,  2.],
        [ 3.,  4.],
        [ 0.,  0.],
        [ 7.,  8.],
        [ 9., 10.]])

Right Context:
+ tensor([[ 3.,  4.],
        [ 5.,  6.],
        [ 0.,  0.],
        [ 9., 10.],
        [11., 12.],
        [ 0.,  0.]])

Samples with Context:
+ tensor([[ 0.,  0.,  1.,  2.,  3.,  4.],
        [ 1.,  2.,  3.,  4.,  5.,  6.],
        [ 3.,  4.,  5.,  6.,  0.,  0.],
        [ 0.,  0.,  7.,  8.,  9., 10.],
        [ 7.,  8.,  9., 10., 11., 12.],
        [ 9., 10., 11., 12.,  0.,  0.]])