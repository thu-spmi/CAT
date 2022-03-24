# CAT（yesno）项目搭建流程

**目录**
* [项目目录结构](#项目目录结构)
* [0.文件准备](#0-文件准备)
* [1.数据准备](#1-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)
* 	* [prepare_data.sh](#prepare_datash)
	* [prepare_dict.sh](#prepare_dictsh)
	* [T.fst & L.fst](#tfst--lfst)
	* [G.fst](#gfst)
	* [TLG.fst](#tlgfst)
* [2.提取FBank特征](#2-%E6%8F%90%E5%8F%96fbank%E7%89%B9%E5%BE%81)
* [3.准备分母图语言模型](#3-%E5%87%86%E5%A4%87%E5%88%86%E6%AF%8D%E5%9B%BE%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)
* [4.神经网络训练准备](#4-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AE%AD%E7%BB%83%E5%87%86%E5%A4%87)
* [5.模型训练](#5-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
* [6.解码](#6-%E8%A7%A3%E7%A0%81)
* [7.实验展示](#7-实验展示)
* [8.结果分析](#8-结果分析)

此文档的目的是让大家了解kaldi工具包的使用，**通过搭建一个简单的语音识别项目，帮助初学者更多了解CAT的工作流程，先知其然，在知其所以然，如果想要更多了解建议进一步阅读以下基本文献。

- L. R. Rabiner, “A tutorial on hidden Markov models and selected applications in speech recognition”, Proceedings of the IEEE, 1989.[PDF](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
- A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber, “Connectionist temporal classiﬁcation: Labelling unsegmented sequence data with recurrent neural networks”, ICML, 2006.[PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- Hongyu Xiang, Zhijian Ou, "CRF-based Single-stage Acoustic Modeling with CTC Topology", ICASSP, 2019.[PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf)
- Zhijian Ou, "State-of-the-Art of End-to-End Speech Recognition", Tutorial at The 6th Asian Conference on Pattern Recognition (ACPR2021), Jeju Island, Korea, 2021.[PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ACPR2021%20Tutorial%20State-of-the-Art%20of%20End-to-End%20Speech%20Recognition.pdf)

**[CAT workflow](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md)已经整理了CAT的工作流程，分为六步，前五步为训练，第六步是解码。** 这份文档将根据[CAT workflow](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md)，更具体地以一个简单语音识别项目（yesno项目）为例，对CAT工作流程加以解释。

yesno语音识别项目，来自[Kaldi中的yesno项目](https://github.com/kaldi-asr/kaldi/tree/master/egs/yesno)。如下所述，yesno项目只含有两个词汇，yes和no；一句话中会包含多个由希伯来语（Hebrew）说的yes和no。

```
The "yesno" corpus is a very small dataset of recordings of one individual
saying yes or no multiple times per recording, in Hebrew.  It is available from
http://www.openslr.org/1.
```

## 项目目录结构

**一个语音识别项目**，指在一个特定的数据集上的项目，通常各个项目在egs文件目录下，也可以尝试训练egs目录下的其它数据集实验。

**yesno**

```
├── cmd.sh #脚本配置
├── path.sh #环境变量配置
├── run.sh #实验主程序
├── conf #配置文件目录
│   ├── decode_dnn.config #解码
│   ├── fbank.conf #fbank提取
│   └── mfcc.conf #mfcc提取
├── ctc-crf -> ../../scripts/ctc-crf #ctc-crf程序
├── exp #模型配置
│   ├── demo #demo模型
│   │   └── config.json #demo模型的训练参数
├── input #输入目录
│   └── lexicon.txt #yesno字典
├── local #存放主程序运行各部分脚本块
│   ├── create_yesno_txt.pl #数据预处理waves.txt(音频ID和对应本地路径)
│   ├── create_yesno_waves_test_train.pl #数据训练开发集划分
│   ├── create_yesno_wav_scp.pl #数据预处理waves.scp（音频ID和对应音频内容）
│   ├── get_word_map.pl #对每个词建立映射
│   ├── prepare_data.sh #数据预处理程序
│   ├── prepare_dict.sh #词典预处理程序
│   ├── score.sh #打分脚本（WER）
│   ├── yesno_decode_graph.sh #fst文件整理打包
│   └── yesno_train_lms.sh #语言模型训练
├── steps -> /myhome/kaldi/egs/wsj/s5/steps #链接到kaldi中同名目录，包含各个训练阶段的子脚本，如特征提取 make_fbank.sh等，此路径软连接到Kaldi所在路径
└── utils -> /myhome/kaldi/egs/wsj/s5/utils #链接到kaldi中同名目录，用于协助处理，如数据复制与验证等
```

接下来我们将利用CAT和yesno数据，一步步搭建一个语音识别项目，再次之前请确保您已经完成了[CAT环境配置](https://github.com/HPLQAQ/CAT-tutorial/blob/master/environment.md)和[CAT的安装](https://github.com/thu-spmi/CAT#Installation)。


## 0. 文件准备

在这部分中，我们先准备好项目所需要的整体框架。

1. 在egs下创建yesno目录

2. 编写以下两个脚本
CAT toolkit: 一般无需修改默认路径即可
Kaldi:路径需要修改到下载好的kaldi根目录下
Data:你的yesno根目录下
   - **path.sh**

     ```shell
     # CAT toolkit
     export CAT_ROOT=../../
     export PATH=$CAT_ROOT/src/ctc_crf/path_weight/build:$PATH
     export PATH=$PWD/ctc-crf:$PATH
     # Kaldi
     export KALDI_ROOT=${KALDI_ROOT:-/myhome/kaldi}
     [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
     export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
     [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
     . $KALDI_ROOT/tools/config/common_path.sh
     export LC_ALL=C
     # Data
     export DATA_ROOT=data/yesno
     ```

     配置全局的环境变量，分别配置CAT、kaldi、Data(数据集的环境变量)，代码来源为`egs\wsj`项目下的同名文件。

     创建完后可以在终端里运行一遍`./path.sh`，没有问题后我们进行下一步。

   - **cmd.sh**

     ```shell
     export train_cmd=run.pl
     export decode_cmd=run.pl
     export mkgraph_cmd=run.pl
     export cuda_cmd=run.pl
     ```

     这里是沿用来自kaldi的并行化工具，适应不同的环境可以配置queue.pl等以及不同的参数。一般情况下我们默认run.pl即可。

3. 创软连接到kaldi以及CAT工具包的目录，便于代码的编写以及迁移

   ```shell
   ln -s ../../scripts/ctc-crf ctc-crf
   ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
   ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
   ```

4. 创建local目录，存放本项目专用数据集，训练，切分，打分等脚本编写

5. 创建**run.sh**，我们在run.sh完成整体编写

   ```shell
   #!/bin/bash
   
   # Copyright 2022 TasiTech
   # Author: Ziwei Li
   # yesno for CAT
   
   # environment
   . ./cmd.sh
   . ./path.sh
   
   #set 
   H=`pwd`  # home dir
   n=12     # parallel jobs=$(nproc)
   stage=0  # set work stages
   stop_stage=9
   change_config=0
   yesno=$DATA_ROOT  #data root
   
   . utils/parse_options.sh
   
   NODE=$1
   if [ ! $NODE ]; then
     NODE=0
   fi
   
   if [ $NODE == 0 ]; then
     if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
       echo "stage 1: *"
       # work
     fi
   
     #more stages
   fi
   ```
   
   $NODE指实验运行的节点数，若运行run.sh时直接传参节点数，用stage和stop_stage控制代码运行部分。

## 1. 数据准备
Step 1: [Data preparation](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md#Data-preparation)

我们完成了框架准备自此进入[CAT workflow](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md)的工作流程，我们按顺序编写每个脚本。

在run.sh中step 1，我们完成以下步骤：获取训练数据，建立所需字典，训练语言模型。

以下为step 1的代码，在本节中我们会详细解释这部分代码的思路。

```shell
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "stage 1: Data Preparation and FST Construction"

  local/prepare_data.sh || exit 1; # Get data and lists
  local/prepare_dict.sh || exit 1; # Get lexicon dict

  # Compile the lexicon and token FSTs
  # generate lexicon FST L.fst according to words.txt, generate token FST T.fst according to tokens.txt
  ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
    data/dict data/local/lang_phn_tmp data/lang || exit 1;
  
  # Train and compile LMs. Generate G.fst according to lm, and compose FSTs into TLG.fst
  local/yesno_train_lms.sh data/train/text data/dict/lexicon.txt data/lm || exit 1;
  local/yesno_decode_graph.sh data/lm/srilm/srilm.o1g.kn.gz data/lang data/lang_test || exit 1;
fi
```

### prepare_data.sh

我们将数据下载准备的步骤放在prepare_data.sh中完成。在prepare.sh完成后，我们期望获得以及划分为训练集(train)与开发集(dev)的data（wav.scp），说话人信息（spk2utt、utt2spk，这里说话人我们默认他为global），标注文本信息（text），分别存储在data/dev,data/train下。

1. 在local目录下创建文件prepare_data.sh，并获取数据

   ```shell
   #!/usr/bin/env bash
   # This script prepares data and create necessary files
   
   . ./path.sh
   
   data=${H}/data
   local=${H}/local 
   mkdir -p ${data}/local
   
   cd ${data}
   
   # acquire data if not downloaded
   if [ ! -d waves_yesno ]; then
     echo "Getting Data"
     wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;
     tar -xvzf waves_yesno.tar.gz || exit 1;
     rm waves_yesno.tar.gz || exit 1;
   fi
   ```

   这一步完成后，我们在data/waves_yesno下得到原始音频数据集。

2. 由于数据且没有划分，这部分我们将音频数据集划分为训练集(train)和开发集(dev)

   注：由于数据量较小这里直接将开发集作为测试集，可以修改

   ```shell
   echo "Preparing train and dev data"
   
   rm -rf train dev
   
   # Create waves list and Divide into dev and train set
   waves_dir=${data}/waves_yesno
   ls -1 $waves_dir | grep "wav" > ${data}/local/waves_all.list
   cd ${data}/local
   ${local}/create_yesno_waves_test_train.pl waves_all.list waves.dev waves.train
   ```
  
  我们完成后生成create_yesno_waves_test_train.pl后我们对其进行编写
   
   **create_yesno_waves_test_train.pl**

   注：这部分代码来源于kaldi中yesno项目

   .pl为perl代码，此部分代码比较难理解。

   ```perl
   #!/usr/bin/env perl
   
   $full_list = $ARGV[0];
   $test_list = $ARGV[1];
   $train_list = $ARGV[2];
   
   open FL, $full_list;
   $nol = 0;
   while ($l = <FL>)
   {
   	$nol++;
   }
   close FL;
   
   $i = 0;
   open FL, $full_list;
   open TESTLIST, ">$test_list";
   open TRAINLIST, ">$train_list";
   while ($l = <FL>)
   {
   	chomp($l);
   	$i++;
   	if ($i <= $nol/2 )
   	{
   		print TRAINLIST "$l\n";
   	}
   	else
   	{
   		print TESTLIST "$l\n";
   	}
   }
   ```

   等分waves_all.list到waves.dev, waves.train中

我们继续回到prepare.data.sh生成test.txt和wave.scp

3. 生成\*_wav.scp, \*.txt(\*代指train, test, dev)

   ```shell
   cd ${data}/local
   
   for x in train dev; do
     # create id lists
     ${local}/create_yesno_wav_scp.pl ${waves_dir} waves.$x > ${x}_wav.scp #id to wavfile
     ${local}/create_yesno_txt.pl waves.$x > ${x}.txt #id to content
   done
   
   ${local}/create_yesno_wav_scp.pl ${waves_dir} waves.dev > test_wav.scp #id to wavfile
   ${local}/create_yesno_txt.pl waves.dev > test.txt #id to content
   ```

生成*.scp文件格式为音频ID和对应的存储位置
  
  **create_yesno_wav_scp.pl**

   创建*.scp文件，内容为文件名对应的存储位置。

   ```perl
   #!/usr/bin/env perl
   
   $waves_dir = $ARGV[0];
   $in_list = $ARGV[1];
   
   open IL, $in_list;
   
   while ($l = <IL>)
   {
   	chomp($l);
   	$full_path = $waves_dir . "\/" . $l;
   	$l =~ s/\.wav//;
   	print "$l $full_path\n";
   }
   ```
生成*.txt文件，未见内容为音频ID和对应的文本内容


   **create_yesno_txt.pl**

   创建.txt文件，内容为文件名对应的文本内容。

   ```perl
   #!/usr/bin/env perl
   
   $in_list = $ARGV[0];
   
   open IL, $in_list;
   
   while ($l = <IL>)
   {
   	chomp($l);
   	$l =~ s/\.wav//;
   	$trans = $l;
   	$trans =~ s/0/NO/g;
   	$trans =~ s/1/YES/g;
   	$trans =~ s/\_/ /g;
   	print "$l $trans\n";
   }
   ```

最后编写prepare_data.sh生成utt2spk，spk2utt
   

4. 将数据转移到data/dev, data/train, data/test下，并生成utt2spk, spk2utt

   ```shell
   for x in train dev test; do
     # sort wave lists and create utt2spk, spk2utt
     mkdir -p $x
     sort local/${x}_wav.scp -o $x/wav.scp
     sort local/$x.txt -o $x/text
     cat $x/text | awk '{printf("%s global\n", $1);}' > $x/utt2spk
     sort $x/utt2spk -o $x/utt2spk
     ${H}/utils/utt2spk_to_spk2utt.pl < $x/utt2spk > $x/spk2utt
   done
   ```

   utils和step目录下的脚本均为kaldi的脚本，在其目录下有详细解释。

   这一流程完成后，data下的目录结构为：
   
   ```
   ├── dev #开发集
   │   ├── spk2utt #说话人-音频ID
   │   ├── text #音频ID-文本
   │   ├── utt2spk #音频ID-说话人
   │   └── wav.scp #音频ID-文件位置
   ├── train #训练集
   │   ├── spk2utt
   │   ├── text
   │   ├── utt2spk
   │   └── wav.scp
   ├── test #测试集
   │   ├── spk2utt
   │   ├── text
   │   ├── utt2spk
   │   └── wav.scp
   ├── local #中间文件
   │   ├── dev.txt #开发集的text
   │   ├── dev_wav.scp #开发集的wav.scp
   │   ├── test.txt
   │   ├── test_wav.scp
   │   ├── train.txt
   │   ├── train_wav.scp
   │   ├── waves.dev 
   │   ├── waves.train #训练集文件名列表
   │   └── waves_all.list
   └── waves_yesno #音频数据集存储位置
   ```

   以下展示train目录下的文件的部分内容：

   **spk2utt**
   
   [speaker] [wav_name1] [wav_name2] ...

   ```
   global 0_0_0_0_1_1_1_1 0_0_0_1_0_0_0_1 0_0_0_1_0_1_1_0 0_0_1_0_0_0_1_0 0_0_1_0_0_1_1_0 0_0_1_0_0_1_1_1 0_0_1_0_1_0_0_0 0_0_1_0_1_0_0_1 0_0_1_0_1_0_1_1 0_0_1_1_0_0_0_1 0_0_1_1_0_1_0_0 0_0_1_1_0_1_1_0 0_0_1_1_0_1_1_1 0_0_1_1_1_0_0_0 0_0_1_1_1_0_0_1 0_0_1_1_1_1_0_0 0_0_1_1_1_1_1_0 0_1_0_0_0_1_0_0 0_1_0_0_0_1_1_0 0_1_0_0_1_0_1_0 0_1_0_0_1_0_1_1 0_1_0_1_0_0_0_0 0_1_0_1_1_0_1_0 0_1_0_1_1_1_0_0 0_1_1_0_0_1_1_0 0_1_1_0_0_1_1_1 0_1_1_1_0_0_0_0 0_1_1_1_0_0_1_0 0_1_1_1_0_1_0_1 0_1_1_1_1_0_1_0
   ```
   
   **utt2spk**
   
   [wav_name] [speaker]
   
   ```
   0_0_0_0_1_1_1_1 global
   0_0_0_1_0_0_0_1 global
   0_0_0_1_0_1_1_0 global
   0_0_1_0_0_0_1_0 global
   0_0_1_0_0_1_1_0 global
   ...
   ```
   
   **wav.scp**
   
   [wav_name] [wav_location]

   ```
   0_0_0_0_1_1_1_1 /myhome/CAT/egs/yesno/data/waves_yesno/0_0_0_0_1_1_1_1.wav
   0_0_0_1_0_0_0_1 /myhome/CAT/egs/yesno/data/waves_yesno/0_0_0_1_0_0_0_1.wav
   0_0_0_1_0_1_1_0 /myhome/CAT/egs/yesno/data/waves_yesno/0_0_0_1_0_1_1_0.wav
   0_0_1_0_0_0_1_0 /myhome/CAT/egs/yesno/data/waves_yesno/0_0_1_0_0_0_1_0.wav
   ...
   ```
   
   **text**
   
   [wav_name] [wav_content]
   
   ```
   0_0_0_0_1_1_1_1 NO NO NO NO YES YES YES YES
   0_0_0_1_0_0_0_1 NO NO NO YES NO NO NO YES
   0_0_0_1_0_1_1_0 NO NO NO YES NO YES YES NO
   0_0_1_0_0_0_1_0 NO NO YES NO NO NO YES NO
   0_0_1_0_0_1_1_0 NO NO YES NO NO YES YES NO
   ...
   ```
   
   通过生成这些固定格式的文件，我们可以方便地使用kaldi的工具来优化工作流程。
   
   当前目录下：
   
   ```
   ├── cmd.sh
   ├── ctc-crf -> ../../scripts/ctc-crf
   ├── data
   │   ├── dev
   │   ├── local
   │   ├── test
   │   ├── train
   │   └── waves_yesno
   ├── local
   │   ├── create_yesno_txt.pl
   │   ├── create_yesno_wav_scp.pl
   │   ├── create_yesno_waves_test_train.pl
   │   └── prepare_data.sh
   ├── path.sh
   ├── run.sh
   ├── steps -> /myhome/kaldi/egs/wsj/s5/steps
   └── utils -> /myhome/kaldi/egs/wsj/s5/utils
   ```

### prepare_dict.sh

在prepare_dict.sh中准备我们此次的词典。

通过这部分代码，我们期待在data/dict下获得经过去重和补充噪音<NOISE>、人声噪声<SPOKEN_NOISE>、未知词<UNK>等的词典lexicon.txt，排序并用数字编号的声学单元units.txt，以及用数字标号的词典，lexicon_numbers.txt。

声学单元的选择有多种，可以是音素phone、英文字母character、汉字、片段wordpiece等。词典（lexicon）的作用是，将待识别的词汇表（vocabulary）中的词分解为声学单元的序列。

1. 由于我们yesno实验所需词典较小，词典保存在input/lexicon.txt中

   ```
   <SIL> SIL #静音silence
   YES Y
   NO N
   ```

2. 编写local/prepare_dict.sh

   ```shell
   #!/bin/bash
   
   # This script prepares the phoneme-based lexicon. It also generates the list of lexicon units
   # and represents the lexicon using the indices of the units. 
   
   dir=${H}/data/dict
   mkdir -p $dir
   srcdict=input/lexicon.txt
   
   . ./path.sh
   
   # Check if lexicon dictionary exists
   [ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;
   
   # Raw dictionary preparation
   # grep removes SIL, perl removes repeated lexicons
   cat $srcdict | grep -v "SIL" | \
     perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
     > $dir/lexicon_raw.txt || exit 1;
   
   # Get the set of units in the lexicon without noises
   # cut: remove words, tr: remove spaces and lines, sort -u: sort and unique
   cut -d ' ' -f 2- $dir/lexicon_raw.txt | tr ' ' '\n' | sort -u > $dir/units_raw.txt
   
   # add noises for lexicons
   (echo '<SPOKEN_NOISE> <SPN>'; echo '<UNK> <SPN>'; echo '<NOISE> <NSN>'; ) | \
    cat - $dir/lexicon_raw.txt | sort | uniq > $dir/lexicon.txt || exit 1;
   
   # add noises and number the units
   (echo '<NSN>'; echo '<SPN>';) | cat - $dir/units_raw.txt | awk '{print $1 " " NR}' > $dir/units.txt
   
   # Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
   utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt
   
   echo "Phoneme-based dictionary preparation succeeded"
   ```

   通过这一脚本的运行后，data目录下会生成一个dict目录如下：

   ```
   ├── dict
   │   ├── lexicon_raw.txt #原词典去重和去非语言学发音
   │   ├── units_raw.txt #lexicon_raw词典中音素去重
   │   ├── lexicon.txt #lexicon_raw词典加入非语言学发音并排序
   │   ├── units.txt #units_raw所有音素标号
   │   └── lexicon_numbers.txt #用units.txt代表词典标号
   ```

   以下展示dict中文件的部分内容：

   **lexicon_raw.txt**

   [word] [unit1] [unit2] ...
   
   ```
   YES Y
   NO N
   ```

   **units_raw.txt**
   
   [unit]
   
   ```
   N
   Y
   ```
   
   **lexicon.txt**
   
   ```
   <NOISE> <NSN> #自认噪声
   <SPOKEN_NOISE> <SPN> #人声噪声
   <UNK> <SPN> #未知词
   NO N
   YES Y
   ```
   
   **units.txt**
   
   [unit] [unit_number]
   
   ```
   <NSN> 1
   <SPN> 2
   N 3
   Y 4
   ```
   
   **lexicon_numbers.txt**
   
   [word] [unit_number1] [unit_number2] ...

   ```
   <NOISE> 1
   <SPOKEN_NOISE> 2
   <UNK> 2
   NO 3
   YES 4
   ```
   
   yesno数据集上人声噪声和自然噪声可以忽略。

### T.fst & L.fst

FST（Finite State Transducers 有限状态转换器）FST常与WFST（Weighted Finite State Transducers 加权有限状态转换器）的称呼混用，与之差异的是WFST在转移路径上附加了权重。安装openfst正是为了使用(W)FST。如下图所示，理论上，一个WFST表示了输入符号序列和输出符号序列的加权关系。

![WFST](assets/WFST.png)


想要了解更多了解以下文献：

[M. Mohri, F. Pereira, and M. Riley, “Speech Recognition with Weighted Finite-State Transducers”, Handbook on Speech Processing and Speech, Springer, 2008.](https://cs.nyu.edu/~mohri/pub/hbka.pdf)

根据发音词典、CTC需要的token，我们生成词典（Lexicon）的L.fst以及CTC生成的T.fst。此处用到我们在prepare_dict.sh中准备好的lexicon.txt, units.txt, lexicon_numbers.txt这3个文件去生成。

```shell
# Compile the lexicon and token FSTs
# generate Lexicon FST L.fst according to words.txt, generate Topology FST T.fst according to tokens.txt
ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
  data/dict data/local/lang_phn_tmp data/lang || exit 1;
```

详见ctc-crf/ctc_compile_dict_token.sh的注释。

***fst文件的可视化，参考[https://www.cnblogs.com/welen/p/7611320.html]、[https://www.dazhuanlan.com/shitou103/topics/1489883]***

通过这一步，脚本依次通过lexicon_numbers.txt, units.txt生成了words.txt, tokens.txt，进而生成了T.fst, L.fst。

**words.txt** （代表了L.fst的output symbol inventory，也就是G.fst的input symbol inventory）

```
<eps> 0 #epsilon，空标签，跳出标签为空
<NOISE> 1
<SPOKEN_NOISE> 2
<UNK> 3
NO 4
YES 5
#0 6 #语言模型G的回退符，确定G.fst
<s> 7 #起始
</s> 8 #结束
```

FST确定化（determinization）是指，对于一个fst图，任意输入序列只对应唯一跳转。消歧符号帮助我们确保我们使用的WFST是确定化的，进一步了解推荐阅读《Kaldi语音识别实战》（作者：陈果果等）第五章。

**tokens.txt** （记录了L.fst的input symbol inventory，也是T.fst的output symbol inventory）

```
<eps> 0
<blk> 1
<NSN> 2
<SPN> 3
N 4
Y 5
#0 6 #G.fst回退符
#1 7 #注：#1,#2为对<SPOKEN_NOISE>和<UNK>的消歧，因为两者都映射到<SPN>
#2 8
#3 9 #sil的消歧
```

为了方便理解，以下通过fstprint展示我们生成的fst文件：

**T.fst**

![T.fst](assets/T.fst.png)


**L.fst**（注：如果L.fst中没有#3的话，则T.fst中#3也没有必要。历史上若使用HMM拓扑，则需要引入SIL unit，每个词汇可接SIL也可以不接，因而L.fst需要#3进行消岐。本例使用CTC拓扑，L.fst不用#3）

![L.fst](assets/L.fst.png)


为方便观察，我们去掉\<NOISE\>,  \<SPOKEN_NOISE\>展示fst生成图，当前：

**words.txt**

```
<eps> 0
NO 1
YES 2
#0 3
<s> 4
</s> 5
```

**tokens.txt**

```
<eps> 0
<blk> 1
N 2
Y 3
#0 4
#1 5
```

**T.fst**

![T.fst(new)](https://github.com/HPLQAQ/CAT/blob/master/assets/T.fst(no%20NOISE).png)


**L.fst**

![L.fst(new)](https://github.com/HPLQAQ/CAT/blob/master/assets/L.fst(no%20NOISE).png)


### G.fst

根据data/train/text、dict/lexicon.txt，生成生成语言模型G.fst。

这部分训练我们通过srilm工具完成，放到local/yesno_train_lms.sh中。

```shell
# Train and compile LMs. Generate G.fst according to lm, and compose FSTs into TLG.fst
    local/yesno_train_lms.sh data/train/text data/dict/lexicon.txt data/lm || exit 1;
```

**yesno_train_lms.sh**

```shell
#!/bin/bash

# To be run from one directory above this script.

. ./path.sh

text=$1
lexicon=$2
dir=$3
for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

#text=data/train/text
#lexicon=data/dict/lexicon.txt
#dir=data/lm
mkdir -p $dir

cleantext=$dir/text.no_oov

# Replace unknown words in text by <UNK>
cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantext || exit 1;

# Count unique words
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts || exit 1;

# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

# note: we probably won't really make use of <UNK> as there aren't any OOVs
cat $dir/unigram.counts  | awk '{print $2}' | ${H}/local/get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_map \
   || exit 1;

# note: ignore 1st field of train.txt, it's the utterance-id.
cat $cleantext | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz \
   || exit 1;

# LM is small enough that we don't need to prune it (only about 0.7M N-grams).

# From here is some commands to do a baseline with SRILM (assuming
# you have it installed).
heldout_sent=3 
sdir=$dir/srilm
mkdir -p $sdir
cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  head -$heldout_sent > $sdir/heldout
cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  tail -n +$heldout_sent > $sdir/train

cat $dir/word_map | awk '{print $1}' | cat - <(echo "<s>"; echo "</s>" ) > $sdir/wordlist

ngram-count -text $sdir/train -order 1 -limit-vocab -vocab $sdir/wordlist -unk \
  -map-unk "<UNK>" -interpolate -lm $sdir/srilm.o1g.kn.gz
# -kndiscount
ngram -lm $sdir/srilm.o1g.kn.gz -ppl $sdir/heldout 
```

取3句计算困惑度，运行结果如下：

```
file data/lm/srilm/heldout: 3 sentences, 24 words, 0 OOVs
0 zeroprobs, logprob= -11.09502 ppl= 2.575885 ppl1= 2.899294
```

srilm工具的使用可以见工具下的README，训练中需要处理的文件存放在data/lm目录下，我们将srilm的训练结果存储在data/lm/srilm下。yesno实验使用1-gram的语言模型的结果，储存到srilm.o1g.kn中，语言模型如下：

**srilm.o1g.km**

```
\data\
ngram 1=7

\1-grams:
-0.9542425	</s>
-99	<NOISE>
-99	<SPOKEN_NOISE>
-99	<UNK>
-99	<s>
-0.3079789	NO
-0.4014005	YES

\end\
```

使用n-gram作为语言模型时，习惯上用以上的arpa格式表示，以上[value] [word]的形式意义为logP(word)=value，画图如下：

**G.fst**

![G.fst](assets/G.fst.png)


### TLG.fst

把以上生成的fst文件进行重组，生成TLG.fst。

```shell
local/yesno_decode_graph.sh data/lm/srilm/srilm.o1g.kn.gz data/lang data/lang_test || exit 1;
```

这部分代码中，我们先将语言模型根据word.txt打包到G.fst中，然后用openfst组合出TLG.fst，用于训练。

**yesno_decode_graph.sh**

```shell
#!/bin/bash 
#

if [ -f path.sh ]; then . path.sh; fi

#lm_dir=$1
arpa_lm=$1
src_lang=$2
tgt_lang=$3

#arpa_lm=${lm_dir}/3gram-mincount/lm_unpruned.gz
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

rm -rf $tgt_lang
cp -r $src_lang $tgt_lang

# Compose the language model to FST
gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   utils/remove_oovs.pl /dev/null | \
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$tgt_lang/words.txt \
     --osymbols=$tgt_lang/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $tgt_lang/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $tgt_lang/G.fst 

# Compose the token, lexicon and language-model FST into the final decoding graph
fsttablecompose $tgt_lang/L.fst $tgt_lang/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstarcsort --sort_type=ilabel > $tgt_lang/LG.fst || exit 1;
fsttablecompose $tgt_lang/T.fst $tgt_lang/LG.fst > $tgt_lang/TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"
rm -r $tgt_lang/LG.fst   # We don't need to keep this intermediate FST
```

到此，我们完成了数据文件的准备以及TLG.fst的生成，TLG.fst画图如下：

![TLG.fst](assets/TLG.png)


	

现在你的data目录结构应该如下：

```
├── dev
│   ├── spk2utt
│   ├── text
│   ├── utt2spk
│   └── wav.scp
├── test
│   ...
├── train
│   ...
├── dict
│   ├── lexicon_numbers.txt
│   ├── lexicon_raw.txt
│   ├── lexicon.txt
│   ├── units_raw.txt
│   └── units.txt
├── lang
│   ├── lexicon_numbers.txt
│   ├── L.fst
│   ├── T.fst
│   ├── tokens.txt
│   ├── units.txt
│   └── words.txt
├── lang_test
│   ├── G.fst
│   ├── lexicon_numbers.txt
│   ├── L.fst
│   ├── T.fst
│   ├── TLG.fst
│   ├── tokens.txt
│   ├── units.txt
│   └── words.txt
├── lm
│   ├── srilm
│   ├── text.no_oov
│   ├── train.gz
│   ├── unigram.counts
│   ├── word.counts
│   └── word_map
├── local
│   ├── dev.txt
│   ├── dev_wav.scp
│   ├── lang_phn_tmp
│   ├── test.txt
│   ├── test_wav.scp
│   ├── train.txt
│   ├── train_wav.scp
│   ├── waves_all.list
│   ├── waves.dev
│   └── waves.train
└── waves_yesno
```

至此我们已经完成yesno项目搭建的90%，再次确认目录下每个文件代表内容。

关于词典文件的说明较为简略，希望进一步了解每一个文件的意义，请阅读[Kaldi Data preparation](https://kaldi-asr.org/doc/data_prep.html)文档。

## 2. 提取FBank特征

Step 2: [Feature extraction](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md#Feature-extraction)

第二步，我们提取波形文件的FBank特征（FBank是Filter Bank的缩写，指音频信号经过短时傅里叶变换，得到幅度谱，再经过一组滤波器组的输出），提取的FBank特征存放在fbank文件夹。

注意在conf目录下建立fbank.conf文件，内容为：

```
--sample-frequency=8000
--num-mel-bins=40
```

分别为音频采样率和滤波器个数，yesno数据集音频采样率为8000，滤波器个数我们取40。

***关于FBank：[https://www.jianshu.com/p/b25abb28b6f8]***

```shell
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "stage 2: FBank Feature Generation"
  #perturb the speaking speed to achieve data augmentation
  utils/data/perturb_data_dir_speed_3way.sh data/train data/train_sp
  utils/data/perturb_data_dir_speed_3way.sh data/dev data/dev_sp
  
  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in train_sp dev_sp; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 1 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;  #filter and sort the data files
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;  #achieve cmvn normalization
  done

  for set in test; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 1 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;  #filter and sort the data files
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;  #achieve cmvn normalization
  done
fi
```

在提取声音文件的特征时，此处使用了将声音进行0.9、1.0、1.1三种变速的操作，在一般识别任务中效果会更好。yesno项目我们不做此操作，此处使用该代码作为演示。相关脚本简要说明如下：

* utils/data/perturb_data_dir_speed_3way.sh：变速脚本

* steps/make_fbank.sh：fbank提取脚本

* utils/fix_data_dir.sh：数据排序与过滤

* steps/compute_cmvn_stats.sh：特征归一化，cmvn是指cepstra mean and variance normalization，即减去均值除以标准差的操作。早期语音识别中提取的音频特征是倒谱，故由此得名。在FBank特征得归一化处理，也沿用了该称呼。

## 3. 准备分母图语言模型

Step 3: [Denominator LM preparation](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md#Denominator-LM-preparation)

在第3步，我们先得到得到训练集中每句话的标签（label）序列，可能用到的标签集（label inventory）保存在units.txt中。然后，通过计算标签序列的语言模型并将其表示成den_lm.fst。最后，由den_lm.fst和标签文件出发，计算出标签序列$l$的对数概率 $logp(l)$，称为path weight。详细的步骤内容见注释。

```shell
data_tr=data/train_sp
data_cv=data/dev_sp

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  #convert word sequences to label sequences according to lexicon_numbers.txt and text files in data/lang_phn
  #the result will be placed in $data_tr/ and $data_cv/
  ctc-crf/prep_ctc_trans.py data/lang/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number
  ctc-crf/prep_ctc_trans.py data/lang/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number
  echo "convert text_number finished"

  # prepare denominator
  ctc-crf/prep_ctc_trans.py data/lang/lexicon_numbers.txt data/train/text "<UNK>" > data/train/text_number
  #sort the text_number file, and then remove the duplicate lines
  cat data/train/text_number | sort -k 2 | uniq -f 1 > data/train/unique_text_number
  mkdir -p data/den_meta
  #generate phone_lm.fst, a phone-based language model
  chain-est-phone-lm ark:data/train/unique_text_number data/den_meta/phone_lm.fst
  #generate the correct T.fst, called T_den.fst
  ctc-crf/ctc_token_fst_corrected.py den data/lang/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
  #compose T_den.fst and phone_lm.fst into den_lm.fst
  fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
  echo "prepare denominator finished"
  
  #calculate and save the weight for each label sequence based on text_number and phone_lm.fst
  path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight
  path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight
  echo "prepare weight finished"
fi
```

## 4. 神经网络训练准备

Step 4: [Neural network training preparation](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md#Neural-network-training-preparation)

不同语音识别项目中，这部分处理差别不大。我们对数据集的的特征进行归一化并和之前计算的path weights一起整合到data/pickle下。

```shell
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  mkdir -p data/all_ark
  
  for set in test; do
    eval data_$set=data/$set
  done

  for set in test cv tr; do
    tmp_data=`eval echo '$'data_$set`

    #apply CMVN feature normalization, calculate delta features, then sub-sample the input feature sequence
    feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$tmp_data/utt2spk scp:$tmp_data/cmvn.scp scp:$tmp_data/feats.scp ark:- \
    | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

    ark_dir=$(readlink -f data/all_ark)/$set.ark
    #copy feature files, generate scp and ark files to save features.
    copy-feats "$feats" "ark,scp:$ark_dir,data/all_ark/$set.scp" || exit 1
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  mkdir -p data/pickle
  #create a pickle file to save the feature, text_number and path weights.
  python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1500 \
      data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/pickle/cv.pickle || exit 1
  python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1500 \
      data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/pickle/tr.pickle || exit 1
fi
```

在stage 5结束后，用`fi`结束最开始```if [ $NODE == 0 ]; then```的大括号，进入到训练部分。

## 5. 模型训练

Step 5: [Model training](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md#Model-training)

此时模型训练需要的所有数据已经准备完成，剩下只需要在exp下创建你的一次实验的文件夹(demo)，建立config.json，此处yesno实验可以将config.json进行修改多次实验：

```json
{
    "net": {
        "type": "LSTM",
        "lossfn": "crf",
        "lamb": 0.01,
        "kwargs": {
            "n_layers": 3,
            "idim": 120,
            "hdim": 320,
            "num_classes": 5,
            "dropout": 0.5
        }
    },
    "scheduler": {
        "type": "SchedulerCosineAnnealing",
        "optimizer": {
            "type_optim": "Adam",
            "kwargs": {
                "lr": 1e-3,
                "betas": [
                    0.9,
                    0.99
                ],
                "weight_decay": 0.0
            }
        },
        "kwargs": {
            "lr_min": 1e-5,
            "period": 5,
            "epoch_max": 30,
            "reverse_metric_direc": true
        }
    }
}
```

net参数设置训练使用的模型，参考ctc_crf/model.py；scheduler参数设置学习的策略，参考ctc_crf/scheduler.py，optimizer参数请阅读[torch.optim][https://pytorch.org/docs/stable/optim.html]相关文档。此处我们采用LSTM模型，学习率衰减使用余弦退火策略。

训练的代码如下：

```shell
PARENTDIR='.'
dir="exp/demo"
DATAPATH=$PARENTDIR/data/

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then

  if [ $change_config == 1 ]; then
    rm $dir/scripts.tar.gz
    rm -rf $dir/ckpt
  fi

  unset CUDA_VISIBLE_DEVICES

  if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
    echo ""
    tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
  elif [ $NODE == 0 ]; then
    echo ""
    echo "'$dir/scripts.tar.gz' already exists."
    echo "If you want to update it, please manually rm it then re-run this script."
  fi

  # uncomment the following line if you want to use specified GPUs
  CUDA_VISIBLE_DEVICES="0"                      \
  python3 ctc-crf/train.py --seed=0             \
    --world-size 1 --rank $NODE                 \
    --batch_size=3                              \
    --dir=$dir                                  \
    --config=$dir/config.json                   \
    --data=$DATAPATH                            \
    || exit 1
fi
```

通过以上代码即可完成模型训练。训练的过程图展示可以在你创建的demo目录下的monitor.jpg中找到。

如果需要重新训练，删除scripts.tar.gz和ckpt文件即可。

## 6. 解码

Step 6: [Decoding](https://github.com/thu-spmi/CAT/blob/master/toolkitworkflow.md#Decoding)

计算测试集中每句话每帧的logits并解码。

```shell
nj=1
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  for set in test; do
    ark_dir=$dir/logits/$set
    mkdir -p $ark_dir
    python3 ctc-crf/calculate_logits.py               \
      --resume=$dir/ckpt/bestckpt.pt                     \
      --config=$dir/config.json                       \
      --nj=$nj --input_scp=data/all_ark/$set.scp      \
      --output_dir=$ark_dir                           \
      || exit 1
  done
fi


if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  for set in test; do
    mkdir -p $dir/decode_${set}
    ln -s $(readlink -f $dir/logits/$set) $dir/decode_${set}/logits
    ctc-crf/decode.sh --stage 1 \
        --cmd "$decode_cmd" --nj 1 --acwt 1.0 --post_decode_acwt 1.0\
        data/lang_${set} data/${set} data/all_ark/${set}.scp $dir/decode_${set}
  done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  for set in test; do
    grep WER $dir/decode_${set}/wer_* | utils/best_wer.sh
  done
fi
```

恭喜你已经完成了你的第一个yesno语音识别项目的搭建，训练和解码过程。

现在你的目录结构应该如下图所示：

```
├── cmd.sh
├── conf
│   ├── decode_dnn.config
│   ├── fbank.conf
│   └── mfcc.conf
├── ctc-crf -> ../../scripts/ctc-crf
├── exp
│   └── demo
├── input
│   └── lexicon.txt
├── local
│   ├── create_yesno_txt.pl
│   ├── create_yesno_waves_test_train.pl
│   ├── create_yesno_wav_scp.pl
│   ├── get_word_map.pl
│   ├── prepare_data.sh
│   ├── prepare_dict.sh
│   ├── score.sh
│   ├── yesno_decode_graph.sh
│   └── yesno_train_lms.sh
├── path.sh
├── run.sh
├── steps -> /myhome/kaldi/egs/wsj/s5/steps
└── utils -> /myhome/kaldi/egs/wsj/s5/utils
```
## 7. 实验展示
	
以下是一次默认训练结果展示：
	
![monitor](https://user-images.githubusercontent.com/99643269/158049088-f21ea54a-66be-43cd-801b-ad05cea6e2b0.png)

识别结果如下：
	
```
%WER 5.83 [ 14 / 240, 1 ins, 13 del, 0 sub ] 
```

识别的详细log在exp/demo/decode_test中。

在训练完成后，请在demo文件夹下自动生成的readme.md文件中对你的这次实验进行记录。

## 8. 结果分析
	
**我们利用`waves_yesno`提供的60条语音数据按照5：5将数据分为训练集和验证集**	
	
**定义合适的学习率、迭代次数，根据梯度下降方法对模型进行分别的训练，并记录每一次训练的平均损失函数值**

以下是进行8次实验的结果对比：
	
![image](https://user-images.githubusercontent.com/99643269/158049903-7fccdc78-8ef7-4b95-af1a-1f213dd96b15.png)

**实验结果可以看出CAT(ctc-crf)要优于ctc，由于yesno实验数据简单生成的语言模型并不复杂所以1-gram要比多阶语言模型效果更好**
	
**也可以尝试修改`exp/demo/config.jsn`中参数尝试多次训练**

**🐱‍🏍**	
