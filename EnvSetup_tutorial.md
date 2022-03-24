# CAT环境配置

[TOC]

对于使用CAT工具包，如果你阅读[CAT的安装](https://github.com/thu-spmi/CAT#Installation)仍觉得比较费解的话，那么本文档是你的一个更好的入手文档。

本文档面向初学者，比较详细地说明为使用CAT所需的环境（操作系统、驱动、工具包、工具等），并对一些容易遇到的问题进行解释，同时提供了一些建议初学者参考的入门学习资料，欢迎进行补充。

## System

CAT需要在Linux下使用，建议选择Ubuntu最新的LTS版本或者通过SSH使用远程服务器。

### 安装Linux系统

测试环境：Ubuntu 20.04 [官方下载][1]

Windows下安装双系统：[教程][2]

> Q：为什么使用Ubuntu？
>
> A：Linux是内核，有多种发行版本，Ubuntu的用户量大，对初学者来说，更方便找到各种问题的解答。

### 使用远程服务器

阮一峰的SSH[教程][3]

### Linux使用

入门推荐阅读：

- 《[快乐的Linux命令行][4]》
- 原版：《[*The Linux Command Line*][5]》

可以参考的网站：

- [菜鸟Linux教程][6]
- [Linux Tutorial][7]

> P.S.：kaldi使用shell编写，对shell的掌握和Linux终端的梳理使用，对提高效率很重要。

## Driver

默认使用NVIDIA显卡训练，需要安装显卡驱动

- 测试环境：NVIDIA RTX2060 Driver version：460.80

以下提供两种安装方式

方法一：打开“软件和更新”->“附加驱动”，选择推荐版本的nvidia-driver驱动安装。

方法二：打开终端，输入

```shell
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

安装完成后重启系统。

检验安装成功并确认驱动版本：打开终端输入`nvidia-smi`

## Cuda

CAT目前仅支持GPU训练，Cuda为使用NVIDIA显卡训练DNN（Deep Neural Network）的必要开发环境，以下介绍Cuda的安装。

1. 下载前检查驱动版本号对应可以使用的CUDA版本：[versions][14]

2. 到[pytorch官网][16]查看pytorch支持的cuda版本号

   测试环境：CUDA Version 10.2

3. 下载安装需要的[cuda][15]版本（推荐runfile方式）并遵循网页上指引安装。

   安装时注意选择取消cuda自带的驱动的安装（取消driver前方√即可）

4. 检查系统cuda版本`cat /usr/local/cuda/version.txt`

   > P.S.：`/usr/local/cuda/`一般是一个链接，链接到`/usr/local/cuda-x.x/`，可以通过file命令查看该链接具体属性。系统中可以安装多个cuda版本，在kaldi安装时可以指定你需要使用的cuda版本，但为方便建议和系统默认版本保持一致。

## Conda

Conda是一个方便的python环境管理工具，使用conda可以更加方便快捷地配置环境，并且在同时开发多个项目时避免环境的冲突，以下介绍conda的安装。

> Q: miniconda和anaconda的区别？
>
> A: anaconda会内置包含更多包和环境，同时占用更大的空间；miniconda则只包含最基础的python和conda包管理工具，优点为。anaconda中有的包在miniconda中都可以通过conda install自行安装，本文档测试使用miniconda，但若空间足够，完全可以选择anaconda。

### 安装

1. 下载conda：

   miniconda: wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh

   anaconda: 在https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/下选择最新版本

   以上为清华源，也可以到[官方文档][17]下载

2. 遵循conda[官方文档][17]的instructions安装，安装结束时会提醒将conda加入环境变量，选择加入。

   以下简单列出安装命令：`sh $your_conda.sh`

3. 使用命令`conda info`查看安装信息，确认安装成功

### 环境配置

1. 配置使用清华源，参考tuna[帮助][18]

2. 取消默认的base环境

   conda安装会自动生成并在每次命令行打开时自动启动base环境，base环境可以作为默认的开发环境使用。但是因为base环境无法删除，使用不当可能导致各种问题，所以不建议使用，可以通过以下命令设置为默认关闭。

   `conda config --set auto_activate_base true`

   也可以通过编辑.condarc添加`auto_activate_base: false`实现

3. 建立您的环境：`conda create -n asrcat python=3`

   asrcat可以改成你想用的名字，python可以具体到小版本，测试使用python=3.9

4. 激活环境：`conda activate asrcat`

4. 按照[pytorch][16]官网说明安装

   以下列出安装命令：`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

5. 运行以下脚本确认安装环境：https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py

### 常用命令

```shell
conda info #查看版本信息
conda create -n $env_name python=$version #创建环境
conda create --name $new_name --copy --clone $old_name #复制环境（用于重命名，如果不删除原环境可以去掉--copy命令）
conda remove --name $env_name --all #删除环境
activate $env_name #激活环境
conda deactivate #退出当前环境
conda env list #显示所有环境
conda clean -a #清理多余包，可通过-h查看具体使用方式，通常-a即可
conda search (--full-name) $package_name #查找包，full-name精确查找
conda install/update/remove (--name $env_name) $package_name
conda list #查看当前环境已安装包
```

## Kaldi

kaldi是一个优秀的语音识别开源工具包，CAT使用kaldi进行特征提取，语言模型训练等部分工作，并采用和kaldi相似的目录结构，以下介绍kaldi安装。

项目地址：[kaldi][19]

1. 下载：git clone https://github.com/kaldi-asr/kaldi.git

2. 按照目录下INSTALL文件指引安装

   > 可能遇到的问题：
   >
   > 1. INSTALL文件中提到make时可以使用-j多线程加速编译，线程数为$(nproc)
   > 2. 编译时kaldi可能不支持最新的g++/gcc版本，可以通过apt-get install g++-*安装需要的版本，通过update-alternatives命令实现默认版本切换，测试使用g++-7.5

3. 检查是否安装成功

   ```shell
   cd egs/yesno/s5
   sh run.sh
   ```

   也可以到`src/bin`目录下`ls`查看是否所有的可执行文件都已编译存在。

## CAT

按照CAT的文档安装：

CAT-v2: https://github.com/thu-spmi/CAT/blob/master/install.md

CAT-v1: https://github.com/thu-spmi/CAT/tree/v1#Installation

注：v1的文档中未提到OpenFST-1.6.7的安装，虽然kaldi已经安装了OpenFST，但是kaldi安装的版本并不适用于CAT的编译，请按v2中的说明进行安装。

## Tools

以下介绍一些工作中常用的工具

### Git

免费开源的版本控制系统，[官网][8]

入门教程：

- 廖雪峰的Git[入门教程][10]（较为简单）
- [Gitbook][11]（更加详细）

命令清单：

- 阮一峰的Git[命令清单][9]
- git --help


### Markdown

Markdown 是一种轻量级标记语言，易于进行文档的编写。

- 菜鸟Markdown[教程][12]
- Markdown简明语法[教程][13]

推荐使用编辑器：

- vscode + extensions: Markdown All in One, Markdown Preview Enhanced 优点：方便自定义，可以直接配合git插件使用
- typora 优点：所见即所得（推荐）

### IDE

- pycharm 优点：集成功能丰富（推荐）
- vscode 优点：自定义强大

### Others

copytranslator 阅读英文文献便捷查词

dolphin 方便的图形界面文件浏览器

chrome 比firefox更好用（雾）的浏览器

P3X-OneNote Linux下的非官方开源OneNote软件

## Questions

如果遇到未说明的问题，您可以：

1. open issue；
2. google/baidu；zhihu；stackoverflow；google scholar；
3. project documents

如有任何问题欢迎在本项目下open issue，感谢您为这份工作的补全提供的贡献。

[1]:https://ubuntu.com/download/desktop	"Ubuntu官网下载"
[2]:https://zhuanlan.zhihu.com/p/101307629	"Ubuntu双系统安装教程，知乎"
[3]:https://github.com/wangdoc/ssh-tutorial	"SSH使用教程，阮一峰"
[4]:https://github.com/billie66/TLCL	"《快乐的Linux命令行》"
[5]:http://linuxcommand.org/	"The Linux Command Line"
[6]:https://www.runoob.com/linux/linux-tutorial.html	"菜鸟教程，Linux教程"
[7]:https://github.com/dunwu/linux-tutorial	"dunwu/linux-tutorial"
[8]:https://git-scm.com/	"Git官网"
[9]:https://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html	"Git常用命令清单，阮一峰"
[10]:https://github.com/numbbbbb/Git-Tutorial-By-liaoxuefeng	"Git入门教程，廖雪峰"
[11]:https://git-scm.com/book/zh/v2	"Gitbook"
[12]:https://www.runoob.com/markdown/md-tutorial.html	"菜鸟教程，Markdown教程"
[13]:https://github.com/Melo618/Simple-Markdown-Guide	"Markdown简明语法教程"
[14]:https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html	"Cuda release notes"
[15]:https://developer.nvidia.com/cuda-toolkit-archive	"Cuda download"
[16]:https://pytorch.org/get-started/locally/	"Pytorch Installation"
[17]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html	"Conda Installation"
[18]: https://mirror.tuna.tsinghua.edu.cn/help/anaconda/	"Tuna Anaconda Help"
[19]: https://github.com/kaldi-asr/kaldi	"Kaldi"
