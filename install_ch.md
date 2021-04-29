# CAT安装介绍

[English英文](install.md)

* [安装依赖](#dependencies)
  * [PyTorch](#pytorch)
  * [Kaldi](#kaldi)
  * [OpenFST](#openfst)
* [安装CAT](#cat)

这是一篇CAT的详细安装说明，其中包括CAT的一些依赖（PyTorch和Kaldi）的安装。

## 安装依赖<a id="dependencies"></a>

CAT主要的依赖工具有两个：PyTorch和Kaldi。一些其他的依赖例如用于读取数据的`h5py`模块可以较为简单地通过`pip`或`pip3`命令安装，因此在本文中不涉及。

我们推荐在`conda`创建的环境下安装CAT及其依赖工具，直接在默认用户环境下安装可能需要root权限。

此外CAT依赖于CUDA计算库，纯CPU方式当前并不支持。

### PyTorch <a id="pytorch"></a>

1. 查看CUDA版本；

   ```shell
   $ whereis cuda
   ```

   输出信息中包括你的CUDA路径，通常是 `/usr/local/cuda`, 基于输出信息的路径，查看CUDA版本信息。

   ```shell
   $ <path to cuda>/bin/nvcc -V
   ```

2. 安装对应的PyTorch版本；

   PyTorch[官方文档](https://pytorch.org/get-started/locally/)中给出了不同方法安装最新稳定版的详细命令。

   如果要安装旧版PyTorch或者当前最新稳定版支持的CUDA版本中没有你的版本，可以在[这个页面](https://pytorch.org/get-started/previous-versions/)内寻找适合你的PyTorch版本。

   CAT当前支持Python3和PyTorch1.1+。更早的Python与PyTorch版本支持请参考CAT v1分支。
   
3. 检查编译安装PyTorch使用的CUDA版本

   ```
   $ python3 -m torch.utils.collect_env
   Collecting environment information...
   PyTorch version: 1.8.1
   Is debug build: False
   CUDA used to build PyTorch: 10.2
   ...
   ```

   

### Kaldi<a id="kaldi"></a>

Kaldi官方安装说明

```shell
$ git clone https://github.com/kaldi-asr/kaldi.git
$ cd kaldi && cat INSTALL
```

对于某些数据集，可能需要额外安装IRSTLM/SRILM/Kaldi LM等工具，为了后续说明方便，将Kaldi的安装目录记为`<path to kaldi>`。

### OpenFST<a id="openfst"></a>

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

4. 编译并安装。如果上一步中指定的安装路径权限不足，则需要root权限执行。

   ```shell
   make && make install
   ```

5. [可选] 删除临时文件夹

   ```shell
   $ cd ../
   $ rm -r tmpfst/
   ```

## CAT<a id="cat"></a>

1. 从GitHub获取源代码，把路径记为`<path to CAT>`；

   ```shell
   $ git clone https://github.com/thu-spmi/CAT
   $ export PATH_CAT=$(readlink -f <path to CAT>)
   $ export PATH_Kaldi=$(readlink -f <path to kaldi>)
   $ export PATH_Openfst=$(readlink -f <path to openfst>)
   ```

2. 把CAT中添加的补丁打包到Kaldi中；

   ```shell
   # 复制补丁文件到Kaldi目录
   $ cp $PATH_CAT/src/kaldi-patch/latgen-faster.cc $PATH_Kaldi/src/bin/
   # 进入目录
   $ cd $PATH_Kaldi/src/bin/
   ```

   修改`Makefile`文件：

   在`BINFILES`文件列表中，添加`latgen-faster`。修改后运行命令编译：

   ```shell
   $ make
   ```

   如果一切正常，kaldi目录下会新增文件`kaldi/src/bin/latgen-faster`。

3. 安装 CTC-CRF 模块；

   ```shell
   # 进入目录
   $ cd $PATH_CAT/src/ctc_crf/
   ```

   CTC-CRF 模块会以python模块的形式安装，因此在安装之前，检查并确保当前使用的python和之后要运行CAT的python是同一个：

   ```shell
   $ which python
   # 或
   $ which python3
   ```
   
   编译并安装
   
   ```shell
   # 如果提示permission denied错误，使用root权限运行
   $ CC=gcc-6 CXX=g++-6 make OPENFST=$PATH_Openfst
   ```
   
4. 尝试导入CTC-CRF模块

   ```shell
   $ python
   # or python3
   $ python3
   >>> import torch
   >>> import ctc_crf_base
   ```

   如果没有错误信息则说明导入正常。CTC-CRF模块是依赖于torch的，因此每次导入`ctc_crf_base`之前需要先导入`torch`。

5. 最后一些小的改动：

   将`CAT/egs/<task>/path.sh`文件中的`KALDI_ROOT=...`修改为你的Kaldi安装目录`<path to kaldi>`。其中`<task>`是ASR任务的名称，例如`wsj`和`swbd`。

   在`CAT/egs/wsj`中，链接kaldi文件夹

   ```shell
   $ cd $PATH_CAT/egs/wsj
   $ ln -snf $PATH_Kaldi/egs/wsj/s5/steps steps
   $ ln -snf $PATH_Kaldi/egs/wsj/s5/utils utils
   ```

6. Enjoy it! :rocket: