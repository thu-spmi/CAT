# 基于JoinAP的多语言/跨语言语音识别
**本文档介绍如何使用JoinAP模型进行多语言/跨语言语音识别的研究，推荐先阅读以下参考资料了解理论知识以及相关细节**：

- Chengrui Zhu, Keyu An, Huahuan Zheng and Zhijian Ou, "Multilingual and crosslingual speech recognition using phonological-vector based phone embeddings", IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 2021. [pdf](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf)
- [THU-SPMI@ASRU2021: 基于音位矢量的多语言与跨语言语音识别，促进多语言信息共享与迁移](https://mp.weixin.qq.com/s?__biz=MzU3MzgyNDMzMQ==&mid=2247484519&idx=1&sn=492cc4e098df0077fc51ecb163d8c8a4&chksm=fd3a8843ca4d015560d9cb3fcfc9e0741c0cd898ad69c7b94b6e092f60ee3e6db3c1f9ccf54d&mpshare=1&scene=1&srcid=0612RqU7DGRZG5XQqg0L2Le1&sharer_sharetime=1655005703359&sharer_shareid=96a0960dd6af6941d3216dad8f2d3a50&key=311fd5318431ff9c5328351edecbba7c5d812fe2ebfc0df6c234172e3cd3b056a5dc35c3c9476a894d7828f7932113f61f420f11bd98bd9f19a18dbbce60d74810202a96eb262756df24294667730f65015d74e3b84a12d358110afd52a3e26cd7bfd692bf4322094d61d031aab32954e42b0043521ae4d7a3ba8b52f177429f&ascene=1&uin=MjI2OTIxNjcxMA%3D%3D&devicetype=Windows+10+x64&version=6209051a&lang=zh_CN&exportkey=AxSPQ4EqXRXSVFCXOPz3zSc%3D&acctmode=0&pass_ticket=5FeYTkI0JWlQDdwbOw%2B90azniyK49b4eF6G1m7lzzoG4aLbog8BRp8ZMiC%2BnfXI5&wx_header=0)

**多语言实验采用基于WFST的解码，训练脚本参考了 [`egs/wsj/exp/asr-ctc-crf-phone`](https://github.com/thu-spmi/CAT/blob/master/egs/wsj/exp/asr-ctc-crf-phone/run.sh) 中的单语训练步骤，多语言脚本放在[`egs/commonvoice`](https://github.com/thu-spmi/CAT/tree/master/egs/commonvoice/run_mc.sh)目录下。**

使用CAT框架训练多语言模型需要准备以下数据：

- 每个语言的**标注数据** ( `wav.scp` 与 `text` )：用来训练声学模型。
- 每个语言的纯**文本数据**：用来训练语言模型。
- **发音词典**：用来构建字母图以及FST解码。
- **音位矢量**：用来JoinAP训练。

**本文档将从以下步骤细化说明前期数据准备和`JoinAP`实验相关的参数配置。** 

- [基于JoinAP的多语言/跨语言语音识别](#基于joinap的多语言跨语言语音识别)
  - [数据获取及预处理](#数据获取及预处理)
  - [发音词典](#发音词典)
  - [音位矢量](#音位矢量)
  - [训练及测试](#训练及测试)
  - [微调](#微调)

## 数据获取及预处理

本文档中实验选择开源的[CommonVoice数据](https://commonvoice.mozilla.org/zh-CN/datasets)作为实验数据，这些开源数据可以直接下载得到。下载好的数据由音频及训练、验证、测试文本构成。下载好数据后，执行命令

```bash
bash local/data.sh /path/to/commonvoice_data -lang [LANGUAGE]
```

处理数据并提取 FBank 特征，其中`[LANGUAGE]`为语言标识，如`en, zh`等。

## 发音词典

由于CommonVoice数据没有提供相应的词典，所以需要实验者自己来生成。

**依赖工具安装说明**

词典的构建依赖于 G2P工具——**[Phonetisaurus G2P](https://github.com/AdolfVonKleist/Phonetisaurus)**

**安装**

```bash
# 使用 CAT/install.sh 脚本
bash install.sh g2p-tool
```

使用该工具可以训练基于 FST 的 G2P 模型，部分语言的已训练好的 G2P-FST 模型可以在下面网站找到：**LanguageNet Grapheme-to-Phoneme Transducers (G2P-FST) [1](https://github.com/uiuc-sst/g2ps) | [2](http://www.isle.illinois.edu/speech_web_lg/data/g2ps/)**

FST 模型位于 `g2ps/models/`，以中文为例
```bash
mkdir demo-g2p && cd demo-g2p
# 下载并解压fst文件
wget https://raw.githubusercontent.com/uiuc-sst/g2ps/master/models/mandarin_2_4_4.fst.gz -O - |
  gunzip >mandarin_2_4_4.fst
# 生成发音词典
export PATH="../../src/bin:$PATH"
echo "你好" >wordlist_zh
phonetisaurus-apply --model mandarin_2_4_4.fst --word_list wordlist_zh
# 输出：
# 你好    n i˨˩˦ h ɑ˨˩˦ w
```

其他语种的词典生成也是类似的。

上面只展示了一个例子，例子中获得了中文词“你好”的发音，而一般的词典构建流程为：

1. 基于训练数据获得词表：

   对于 word-based 语言（如英语）获取全部独立词即可；而对 character-based 语言（如中文），可以有不同的方式，例如直接构建字典，或先通过分词工具分词，再按照 word-based 语言处理方式处理获得词表。

2. 获取目标语言的 G2P `.fst` 模型；

3. 基于词表和 fst 模型生成词典。
   ```bash
   export PATH="../../src/bin:$PATH"
   phonetisaurus-apply --model [lang].fst --word_list wordlist_[lang]
   ```

完整流程对应的脚本位于`egs/commonvoice/local/prep_ipa_lexicon.sh`，使用

```bash
# 在 egs/commonvoice/ 下运行
bash local/prep_ipa_lexicon.sh -h
```

查看帮助信息。输出文件默认位于`data/lang-${lang}`。

## 音位矢量

在多语言声学模型训练时，为了促进多语言信息共享与迁移，[JoinAP论文](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf)引入音位矢量（phonological-vector）来表示每个音素。音位矢量的构建用到了panphon工具包。panphon工具包定义了全部 IPA 音素符号到发音特征（Articulatory Feature, AF）的映射；这样可以根据 IPA 音素得到它的发音特征表达，进而编码成51维音位矢量（描述见后）。

在传统方法中定义了帧t在通过声学声学深度神经网络（Deep Neural Network、DNN）提取出高层声学特征 $h_{t}$ ，最后经过 linear layer 计算 logit，如下所示：
$z_{t,i} = e_{i}^{T} h_{t}$ 其中 $e_{i}$ 表示为音素 $i$ 的 `phone embadding` ，其具体值为 linear layer 的权重向量。

而在 JoinAP（Joining of Acoustics and Phonology）方法中，$e_{i}$ 并不是linear layer的权重向量，它是由 $p_{i}$ （51维的音位矢量，具体生成方式如下面IPA2AF映射所示）经过线性或非线性变换得到：

- The JoinAP-Linear method

  $e_{i} = Ap_i$ 其中A为线性层的权重参数，由模型学习生成

- The JoinAP-Nonlinear method

  $e_{i} = A_2\sigma(A_{1}p_{i})$ 其中 $A_{1}$ 和 $A_{2}$ 为非线性层的权重参数，由模型学习生成，$ \sigma $ 为激活函数

最后将音素 $i$ 的 `phone embedding` 与声学特征 $h_{t}$ 做内积，计算出 $t$ 时刻下音素 $i$ 的匹配得分（logit），便可用于基于 CTC 或 CTC-CRF 的语音识别。不难看出，JoinAP方法引入音位矢量，对声学神经网络的最后输出线性层进行了修改。

<p align="center">
  <img width="200" src="../assets/JoinAP.png" alt="JoinAP">
</p>

**[panphon工具包](https://github.com/dmort27/panphon)**

我们需要对每个音素单元进行标记以得到其音位矢量。panphon一共提供24个发音特征（AF），每种发音特征分别有 “+”、“-”、“0” 三种取值；我们将**其中 “+” 被编码 “10”，“-”被编码为 “01”，“00” 则表示 “0” 符号**。这样，24维的发音特征被编码为了 48 维的矢量；再加上对三个特殊单元：blk（空）、spn（说话噪音）、nsn（自然噪音）的3维编码，便得到51维音位矢量。

**[panphon提供的IPA音素到发音特征的映射表(IPA2AF)](https://github.com/dmort27/panphon/blob/master/panphon/data/ipa_all.csv)**

我们可以通过IPA2AF映射表对每个音素进行编码，得到音位矢量。以下展示以德语为例：

| token ID | IPA | syl+ | syl- | son+ | son- | cons+ | cons- | cont+ | cont- | delrel+ | delrel- | lat+ | lat- | nas+ | nas- | srtid+ | strid- | voi+ | voi- | sg+ | sg- | cg+ | cg- | ant+ | ant- | cor+ | cor- | distr+ | distr- | lab+ | lab- | hi+ | hi- | lo+ | lo- | back+ | back- | round+ | round- | velaric+ | velaric- | tense+ | tense- | long+ | long- | hitone+ | hitone- | hireg+ | hireg- | blk | nsn | spn |
|:-------- |:--- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ----- | ------- | ------- | ---- | ---- | ---- | ---- | ------ | ------ | ---- | ---- | --- | --- | --- | --- | ---- | ---- | ---- | ---- | ------ | ------ | ---- | ---- | --- | --- | --- | --- | ----- | ----- | ------ | ------ | -------- | -------- | ------ | ------ | ----- | ----- | ------- | ------- | ------ | ------ | --- | --- | --- |
| 0        | BLK | 0    | 0    | 0    | 0    | 0     | 0     | 0     | 0     | 0       | 0       | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0     | 0     | 0      | 0      | 0        | 0        | 0      | 0      | 0     | 0     | 0       | 0       | 0      | 0      | 1   | 0   | 0   |
| 1        | NSN | 0    | 0    | 0    | 0    | 0     | 0     | 0     | 0     | 0       | 0       | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0     | 0     | 0      | 0      | 0        | 0        | 0      | 0      | 0     | 0     | 0       | 0       | 0      | 0      | 0   | 1   | 0   |
| 2        | SPN | 0    | 0    | 0    | 0    | 0     | 0     | 0     | 0     | 0       | 0       | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0     | 0     | 0      | 0      | 0        | 0        | 0      | 0      | 0     | 0     | 0       | 0       | 0      | 0      | 0   | 0   | 1   |
| 3        | #   | 0    | 0    | 0    | 0    | 0     | 0     | 0     | 0     | 0       | 0       | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0     | 0     | 0      | 0      | 0        | 0        | 0      | 0      | 0     | 0     | 0       | 0       | 0      | 0      | 0   | 0   | 0   |
| 4        | 1   | 0    | 0    | 0    | 0    | 0     | 0     | 0     | 0     | 0       | 0       | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0     | 0     | 0      | 0      | 0        | 0        | 0      | 0      | 0     | 0     | 0       | 0       | 0      | 0      | 0   | 0   | 0   |
| 5        | 7   | 0    | 0    | 0    | 0    | 0     | 0     | 0     | 0     | 0       | 0       | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0      | 0      | 0    | 0    | 0   | 0   | 0   | 0   | 0     | 0     | 0      | 0      | 0        | 0        | 0      | 0      | 0     | 0     | 0       | 0       | 0      | 0      | 0   | 0   | 0   |
...

在论文增加了3个特殊符号 `BLK, NSN, SPN`，当前实验中，我们仅使用`<blk>， <unk>`。

IPA映射矩阵在调用`local/prep_ipa_lexicon.sh`脚本时已经一并生成，也可以重新生成该 IPA 映射矩阵：

```bash
python local/get_ipa_mapping.py -h
```
> **注意：** 映射表中未出现的音素，称之为集外音素。对于作为分隔符号或停顿语气等对训练无影响的音素可以直接全部标记为0；其它集外音素将其映射到与其它声学上最相似的音素。

输出文件为`np.array`格式，读取方式为：

```python
import numpy as np
de=np.load('de.npy')
'''
array( [[0, 0, 0, ..., 1, 0, 0],   
        [0, 0, 0, ..., 0, 0, 1],
           ...,
        [1, 0, 1, ..., 0, 0, 0],
        [0, 1, 0, ..., 0, 0, 0]], dtype=int64)
'''
```

至此，完成音位矢量的构建，具体流程可概括如下：

![pv.feature](../assets/phonological_feature.png)

## 训练及测试

在v3版本的CAT中使用`config.json` 和 `hyper-p.json` 两个配置文件控制整个模型的配置，其中`config.json`文件用来配置模型参数， `hyper-p.json` 文件用来配置一些超参数，具体可以参考[CAT官方配置指南](./configure_guide.md)。
我们只需要修改`config.json`文件的`encoder`里的参数来对JoinAP模型进行设置。

```
{
    "encoder": {
        "type": "JoinAPLinearEncoder",
        "kwargs": {
            "pv_path": "data/de.npy",
            "enc_head_type": "ConformerNet",
            ...
        }
    },
}
```
其中 `type` 参数支持 `JoinAPLinearEncoder` 和 `JoinAPNonLinearEncoder` ，分别对应 `JoinAP` 论文中的线性和非线性方法。  
`pv_path` 是上一步准备的`numpy`格式的音位矢量文件。`enc_head_type` 是采用的模型类型，如 `Conformer, LSTM, TDNN` 等。其余参数由`enc_head_type`决定，目前支持的模型类型和相关参数可以查看 [源码](../cat/shared/encoder.py)

## 微调

多语言模型微调时，只需要使用多语模型初始化单语模型参数，其他训练步骤跟单语保持不变。但是因为多语言参数跟单语模型不匹配，需要在初始化前作修改。从多语模型到单语模型，最主要的改变在以下两点：

- token数量
- JoinAP方法中音位矢量的维度（因为跟 token 数量有关）

所以需要修改有关这两个变量的参数。使用脚本[unpack_mulingual_param.py](../egs/commonvoice/exp/joinap/unpack_mulingual_param.py)从多语模型中获取单语模型参数。

**实验参数配置**

在CAT工具包使用多语言预训练模型在目标语言单语数据上微调ASR模型，需要如下三个步骤。其他部分与 [`egs/wsj/exp/asr-ctc-crf-phone`](https://github.com/thu-spmi/CAT/blob/master/egs/wsj/exp/asr-ctc-crf-phone/run.sh) 中单语训练步骤一致。

**注意**

在单语数据上微调时，加载预训练模型有两种方式：

1. `train:option`中设置`init_model`参数为对应模型参数，该方式下只会导入模型参数，scheduler、optimizer等是重新初始化的；
2. `train:option`中设置`resume`参数为对应模型参数，该方式等效于在预训练模型上“接着”训练，模型、scheduler以及optimizer都是继承的，因此该方式要求单语和多语模型参数严格相同。


**至此我们完成基于JoinAP的多语言/跨语言语音识别实验的全部步骤！**

完整的实验示例参考[commonvoice](../egs/commonvoice/exp/joinap/readme.md)

✈-🐱‍🏍
