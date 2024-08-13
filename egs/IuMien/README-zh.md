Author:
<br>Fanhua Song
<br>Lukuan Dong 330293721@qq.com

[[English]](./README.md)[[中文]](./README-zh.md)

### Note

 This is the official code for the paper "[Low-Resourced Speech Recognition for Iu Mien Language via Weakly-Supervised Phoneme-based Multilingual Pre-training](https://arxiv.org/abs/2407.13292)". 


### 任务：基于音素的预训练模型在以勉语为例的汉藏语系的跨语言语音识别研究

主流的自动语音识别（ASR，automatic speech recognition）技术，通常需要数百到数千小时的带标注语音数据。对于低资源语音识别，基于音素监督预训练、基于子词监督预训练、自监督预训练，是三种常用多语言预训练方法。勉语是中国瑶族的主要民族语言，由于带标注的语音数据非常有限，勉语属于低资源语言。我们在***不到10小时的勉语转录语音标注数据***上，研究并比较了这三种方法用于勉语语音识别的效果。我们的实验基于最近发布的三种预训练模型，这些模型是在CommonVoice数据集的10种语言（CV-Lang10），共计4096小时上预训练的，分别对应于低资源ASR的三种方法。研究发现，音素监督取得了比子词监督和自监督更好的性能，表现出数据高效性。***[Whistle](https://arxiv.org/abs/2406.02166)（weakly-supervised phoneme-based multilingual pre-training）模型***，即通过弱监督的基于音素的多语言预训练获得的模型，在测试集上取得了最好的结果。


### 数据

[勉语语料](https://github.com/mightmay/MightLJSpeech/tree/master/MightLJSpeech-1.1/csv)（瑶语的一种，和金秀瑶语相近。总共9754条数据，大概9.7小时的有标签音频数据）

该数据集文本领域来自于圣经，使用标准勉文方案进行书写。

标准勉文方案（参考自[勉语维基百科](https://zh.wikipedia.org/wiki/%E5%8B%89%E6%96%B9%E8%A8%80)）只使用基本拉丁字母的26个字母作为书写的基本单位，构建出30个声母、128个韵母和8个声调。

值得注意的是，勉语共有8种音调，勉语单词拼写会显式的写成单词音调，一般单词的拼写最后一位表示音调（但是1声不显式写出），需要注意的是音调塞音韵尾只能配合两个入声调，其他韵尾只能配合舒声调。入声调原来用单独调号q和r，现在用相近的舒声调的调号（v和c）

勉文文本举例：
```
Psalm_9_39_189040_195800	ninh mbuo nyei zaux yaac zuqc ninh mbuo ganh zaeng nyei mungz nzenc jienv
Psalm_9_41_199200_201880	ninh baengh fim nyei siemv zuiz
Psalm_9_43_208440_213920	orqv mienh se la kuqv tin hungh nyei fingx fingx mienh
Psalm_9_44_213920_217480	zungv zuqc mingh yiem yiemh gen
Psalm_9_45_217480_223040	ninh maiv zeiz yietc liuz la kuqv dangx donx nyei mienh
Psalm_9_47_229960_235040	o ziouv aah daaih maah maiv dungx bun baamh mienh duqv hingh
```

 
### 数据处理
#### 下载数据
```shell
git clone https://github.com/mightmay/MightLJSpeech.git MightLJSPeech
```
#### 数据划分

按8:1:1的比例划分训练集、验证集、测试集
对于验证集和测试集，删除了其与训练集文本完全重复的音频及其对应文本。

```shell
python utils/split_data.py
python utils/fliter_data.py
```

我们实验中所使用的训练集、验证机、测试集划分，可以参考[exp_data](./exp_data/)文件夹

#### 音频特征提取

因为本实验所使用的预训练模型在训练时使用的输入fbank音频特征由kaldi提取得到，为了与预训练模型对齐，我们也基于kaldi提取fbank音频特征

```shell
bash uitils/data_kaldi.sh
python utils/data/resolvedata.py
```

对于使用自监督模型wav2vec的实验，因为其输入并非fbank特征，而是16khz采样率的原始音频信号，使用以下脚本处理数据。

```shell
bash local/audio2ark.sh
```

### 发音词典构建

参考勉文的维基百科中关于勉文拼写与IPA发音的[对照表](https://zh.wikipedia.org/wiki/%E5%8B%89%E6%96%B9%E8%A8%80#:~:text=%E6%B1%89%E8%AF%AD%E6%8B%BC%E9%9F%B3%E5%BE%88%E5%83%8F%E3%80%82-,IMUS%E6%8B%BC%E5%86%99%E5%AF%B9%E5%BA%94,-%5B%E7%BC%96%E8%BE%91%5D)，使用训练集生成词表，对词表中的词按最长匹配规则切分后，参考对照表得到单词的IPA发音标注词典。我们实验中使用的发音词典请参考`exp_dict/`。

```shell
python utils/get_wordlist.py
python utils/get_lexicon.py
```

### 语言模型

在语音识别中，我们经常使用语言模型来辅助解码，以此来降低最终识别结果的错误率。在本实验中，我们使用4-gram-word-lm作为我们使用的语言模型。

语言模型训练方法
```shell
bash exp/decode_lm/run.history.sh
```


### 模型结构

实验使用Conformer结构的Encoder，使用vgg21网络对音频信号进行降采样，共14层Conformer block，每层4个注意力头，卷积核大小为15，hidden_dim为512


### 实验结果

取三次独立实验的均值作为最终的实验结果。

（1）bpe建模，使用勉语数据从零开始训练

| model                            | Unit | lm setting   | test | note |
| -------------------------------- | ---- | ---- | -----  | ---- |
| [Mono-subword](exp/Mono-subword/) | bpe500  | no lm | 9.71 | |
| [Mono-subword](exp/Mono-subword/) | bpe500  | 4-gram word lm  | 6.87 | use fst decode |

（2）bpe建模，在十国子词预训练模型的基础上使用勉语数据进行微调

| model                            | Unit | lm setting  | test | note |
| -------------------------------- | ---- | ---- | -----  | ---- |
| [Mul10-sub-PT-sub-FT](exp/Mul10-sub-PT-sub-FT/) | bpe500  | no lm  | 4.33 | |
| [Mul10-sub-PT-sub-FT](exp/Mul10-sub-PT-sub-FT/) | bpe500  | 4-gram word lm | 3.46 | use fst decode |

(3) bpe建模，在Wav2vec2-cv10预训练模型的基础上使用勉语数据进行微调

| model                            | Unit | Evaluation metrics | lm setting  | test | note |
| -------------------------------- | ---- | ---- | ----- | --- | ---- | 
| [Wav2vec2-cv10-sub-FT](exp/Wav2vec2-cv10-sub-FT/) | bpe500  | wer | no lm | 3.76 | |
| [Wav2vec2-cv10-sub-FT](exp/Wav2vec2-cv10-sub-FT) | bpe500  | wer  | 4-gram word lm | 3.06 | | 

（4）bpe建模，在Whistle-small预训练模型的基础上使用勉语数据进行微调

| model                            | Unit | lm setting   | test | note |
| -------------------------------- | ---- | ---- | -----  | ---- |
| [Whistle-sub-FT](exp/Whistle-sub-FT/) | bpe500  | no lm  | 3.30 | |
| [Whistle-sub-FT](exp/Whistle-sub-FT) | bpe500  | 4-gram word lm  | 2.95  | use fst decode |

（5）phone建模，使用勉语数据从零开始训练

| model                            | Unit | Evaluation metrics | lm setting  | test | note |
| -------------------------------- | ---- | ---- | ----  | ---- | ---- |
| [Mono-phoneme](exp/Mono-phoneme/) | phone  | per | no lm | 4.22 | |
| [Mono-phoneme](exp/Mono-phoneme) | phone  | wer |4-gram word lm  | 4.69 | |


（6）phone建模，在Whistle-small预训练模型的基础上使用勉语数据进行微调

| model                            | Unit | Evaluation metrics | lm setting  | test | note |
| -------------------------------- | ---- | ---- | ----- | ----- | ---- |
| [Whistle-phoneme-FT](exp/Whistle-phoneme-FT/) | phone  | per | no lm  | 2.41 | |
| [Whistle-phoneme-FT](exp/Whistle-phoneme-FT/) | phone  | wer | 4-gram word lm  | 2.71 | |

（7）phone建模，在Wav2vec2-cv10预训练模型的基础上使用勉语数据进行微调

| model                            | Unit | Evaluation metrics | lm setting  | test | note |
| -------------------------------- | ---- | ---- | ----- | ----- | ---- |
| [Wav2vec2-cv10-phoneme-FT](exp/Wav2vec2-cv10-phoneme-FT/) | phone  | per | no lm | 2.53 | |
| [Wav2vec2-cv10-phoneme-FT](exp/Wav2vec2-cv10-phoneme-FT) | phone  | wer | 4-gram word lm  | 2.76 | |

