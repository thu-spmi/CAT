# CAT安装介绍

这是一篇主要面向初学者的详细安装说明，其中包括CAT的一些依赖（PyTorch和Kaldi）的安装。

## 安装依赖

CAT主要的依赖工具有两个：PyTorch和Kaldi。一些其他的依赖例如用于读取数据的`h5py`模块可以较为简单地通过`pip`或`pip3`命令安装，因此在本文中不涉及。

我们推荐在`conda`创建的环境下安装CAT及其依赖工具，直接在默认用户环境下安装可能需要root权限。

此外CAT依赖于CUDA计算库，纯CPU方式当前并不支持。

### PyTorch 

1. 查看CUDA版本；

   ```shell
   $ whereis cuda
   ```

   输出信息中包括你的CUDA路径，通常是 `/usr/local/cuda`, 基于输出信息的路径，查看CUDA版本信息。

   ```shell
   $ cat <path to CUDA>/version.txt
   ```

2. 安装对应的PyTorch版本；

   PyTorch[官方文档](https://pytorch.org/get-started/locally/)中给出了不同方法安装最新稳定版的详细命令。

   如果要安装旧版PyTorch或者当前最新稳定版支持的CUDA版本中没有你的版本，可以在[这个页面](https://pytorch.org/get-started/previous-versions/)内寻找适合你的PyTorch版本。

   CAT当前支持Python2和Python3，对应支持的PyTorch版本为0.4+和1.0+。

### Kaldi

Kaldi的安装请参考详细的安装说明: http://jrmeyer.github.io/asr/2016/01/26/Installing-Kaldi.html

其中IRSTLM工具是必要的，为了后续说明方便，将Kaldi的安装目录记为`<path to kaldi>`。

### OpenFST

1. 创建一个临时文件夹并进入；

   ```shell
   $ mkdir tmpfst && cd tmpfst
   ```

2. 下载OpenFST-1.6.7源文件并解压；

   ```shell
   # fetch the file from source
   $ wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz
   # extract
   $ tar -xf openfst-1.6.7.tar.gz
   # change directory to extracted one
   $ cd openfst-1.6.7/
   ```

3. 配置编译，如果不添加`--prefix`选项，默认的安装路径是`/usr/local`。推荐指定个人目录下的路径。

   ```shell
   $ ./configure --prefix=<path to openfst>
   ```

4. 编译并安装。如果上一步中指定的安装路径不在当前用户所有的目录下，需要root权限执行以下命令。

   ```shell
   make && make install
   ```

5. [可选] 删除临时文件夹

   ```shell
   $ cd ../..
   $ rm -r tmpfst/
   ```

## 安装CAT

1. 从GitHub获取源代码，把路径记为`<path to CAT>`；

   ```shell
   $ git clone https://github.com/thu-spmi/CAT
   ```

2. 把CAT中添加的补丁打包到Kaldi中；

   ```shell
   # 复制补丁文件到Kaldi目录
   $ cp <path to CAT>/src/kaldi-patch/latgen-faster.cc <path to kaldi>/src/bin/
   # 进入目录
   $ cd <path to kaldi>/src/bin/
   ```

   修改`Makefile`文件：

   在`BINFILES`文件列表中，添加`latgen-faster`。修改后运行命令编译：

   ```shell
   $ make
   ```

   如果一切正常，`<path to kaldi>/src/bin/`目录下会新增两个文件：`latgen-faster` 和 `latgen-faster.o`。

3. 安装 CTC-CRF 模块；

   ```shell
   # 进入目录
   $ cd <path to CAT>/src/ctc_crf/
   ```

   CTC-CRF 模块会以python模块的形式安装，因此在安装之前，检查并确保当前使用的python和之后要运行CAT的python是同一个：

   ```shell
   # python2
   $ which python
   # python3
   $ which python3
   ```

   对于Python2，运行以下命令安装即可。安装完成后跳到步骤4；

   ```shell
   # 如果提示permission denied错误，使用root权限运行
   $ make OPENFST=<path to openfst>
   ```

   对于Python3，首先打开`Makefile`文件，注释最后一行命令，修改后的文件类似这样：

   ```makefile
   ...
   CTCCRF: GPUCTC GPUDEN PATHWEIGHT
   #	python setup.py
   ```

   编译CTC-CRF模块的依赖文件：

   ```shell
   $ make OPENFST=<path to openfst>
   ```

   最后，安装CTC-CRF模块

   ```shell
   # 如果提示permission denied错误，使用root权限运行
   $ python3 setup_1_0.py install
   ```

4. 尝试导入CTC-CRF模块

   ```shell
   $ python
   # or python3
   $ python3
   >>> import torch
   >>> import ctc_crf_base
   ```

   如果没有错误信息则说明导入正常。CTC-CRF模块是依赖于torch的，因此在导入`ctc_crf_base`之前需要先导入`torch`。

5. 最后一些小的改动：

   将`<path to CAT>/scripts/ctc-crf/kaldi_io.py`和`<path to CAT>/egs/<task>/path.sh`两个文件中的`KALDI_ROOT=...`修改为你的Kaldi安装目录`<path to kaldi>`。其中`<task>`是ASR任务的目录，例如`wsj`和`swbd`。

6. Enjoy it! :rocket: