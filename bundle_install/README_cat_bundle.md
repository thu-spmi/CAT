# CAT 离线源码安装包

这组脚本用于生成并使用一个不依赖 `git clone` 的 CAT 安装包。目标机仍然可以通过 conda/pip 镜像安装 PyTorch、torchaudio、tensorboard 等普通包；容易被原始安装脚本从 GitHub 或源码站拉取的源码依赖会放进 bundle。

## 1. 在联网机器上构建安装包

如果脚本目录、CAT 仓库目录、bundle 输出目录不在同一个位置，可以显式指定：

```bash
CAT_REPO_ROOT=/path/to/CAT \
OUT_DIR=/path/to/bundles \
bash /path/to/scripts/build_cat_install_bundle.sh
```

如果脚本就在 CAT 仓库内，或当前工作目录是 CAT 仓库，`CAT_REPO_ROOT` 通常可以自动推断；`OUT_DIR` 默认是当前工作目录。
如果 `build_cat_install_bundle.sh`、`install_cat_bundle.sh`、`README_cat_bundle.md` 不在同一个目录，可以额外指定 `INSTALL_SCRIPT=/path/to/install_cat_bundle.sh` 和 `BUNDLE_README=/path/to/README_cat_bundle.md`。

生成结果类似：

```text
/path/to/bundles/cat-install-bundle-20260622-120000.tar.gz
```

可选项：

```bash
# 同时下载普通 Python wheel 到 wheelhouse，适合同 Python/平台复用
DOWNLOAD_PY_WHEELS=1 bash /path/to/scripts/build_cat_install_bundle.sh

# 如果需要缩小包体积，可以剥离依赖源码里的 .git 元数据
STRIP_GIT_METADATA=1 bash /path/to/scripts/build_cat_install_bundle.sh

# 换 pip 下载镜像
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash /path/to/scripts/build_cat_install_bundle.sh

# 生成 latest PyTorch/Transformers profile 的 bundle，不再把 transformers 限制在 <4.54
LATEST_TORCH_TRANSFORMERS=1 bash /path/to/scripts/build_cat_install_bundle.sh
```

## 2. 目标服务器要求

目标服务器不需要能访问 GitHub，也不会执行 `git clone`；但默认安装仍需要访问 conda/pip 镜像来安装 PyTorch、tensorboard 等普通包。安装前建议确认：

- Linux x86_64 服务器，具备 `bash`、`tar`、`gzip`、`sed`、`awk` 等常见基础命令。
- 能访问配置的 conda/pip 镜像；如果设置 `PIP_OFFLINE=1`，需要 bundle 内已经准备好对应 Python/平台可用的 wheelhouse。
- 有可用的 NVIDIA GPU，且 `nvidia-smi` 能正常看到显卡和驱动。默认安装会构建 `ctc-crf` 等 CUDA 扩展，CPU-only 目标机需要显式跳过相关模块。
- 有 CUDA 编译器 `nvcc`。可以使用系统 CUDA，例如 `/usr/local/cuda/bin/nvcc`；如果目标机没有系统 `nvcc`，可尝试设置 `INSTALL_CUDA_BUILD_TOOLKIT=1` 让脚本从 conda 安装 CUDA 编译工具。
- 有可用的 C/C++ 编译器。脚本会通过 conda 安装 CMake、Make、Boost、Eigen 等构建依赖，并在需要 CUDA native build 时默认安装 conda GCC/G++ wrapper；如果系统默认 GCC 过新，例如 CUDA 12.0 遇到 GCC 13，脚本会自动改用兼容的 host compiler。
- `CONDA_HOME` 和 `CAT_INSTALL_ROOT` 对当前用户可写。默认分别为 `~/miniconda3` 和 `~/CAT`。
- 建议至少预留 10GB 可用磁盘空间；PyTorch/CUDA conda 环境和 native 编译中间产物会占用较多空间。
- 如果安装 `fst-decoder` 或 `all`，需要额外提供可用的 Kaldi，并设置 `KALDI_ROOT=/path/to/kaldi`。

可用下面的命令做快速自检：

```bash
uname -m
nvidia-smi
which gcc g++ || true
nvcc --version || true
df -h "$HOME"
```

## 3. 在目标机上安装

```bash
tar -xzf cat-install-bundle-*.tar.gz
cd cat-install-bundle-*
bash install_cat_bundle.sh
```

模块化安装保持原 `install.sh` 的语义：

```bash
# 默认等价于原来的 install.sh cat
bash install_cat_bundle.sh

# 只安装指定模块
bash install_cat_bundle.sh ctcdecode
bash install_cat_bundle.sh kenlm
bash install_cat_bundle.sh ctc-crf

# 安装 CAT 主体模块：ctcdecode + kenlm + ctc-crf + requirements + CAT
bash install_cat_bundle.sh cat

# 全部模块：cat + fst-decoder + g2p-tool
KALDI_ROOT=/opt/kaldi bash install_cat_bundle.sh all
```

默认行为：

- 安装或复用 conda；优先检查 `CONDA_HOME`，再检查 PATH 里的已有 conda。
- 创建或复用 `cat` conda 环境。
- 使用清华 conda/pip 镜像。
- conda 安装 `pytorch==2.1.0`、`torchaudio==2.1.0`、`pytorch-cuda=12.1`。
- 从本地源码安装 `ctcdecode`、`kenlm`、`ctc_crf`、`torch-gather`、`warp-rnnt`、`webdataset`、`warp-ctct`、`ctc-align-cuda`。
- `torch-gather`、`warp-rnnt`、`webdataset`、`warp-ctct`、`ctc-align-cuda` 会先复制到 `CAT/src/` 下，再从该位置安装；editable 安装后的源码路径不会指向解压包的 `payload/`。
- 安装 PyTorch CUDA 扩展时，如果目标机的 CUDA runtime 只有 `libcudart.so.12` 这类版本化文件，或位于 `/usr/lib/x86_64-linux-gnu` 等 multiarch 目录，脚本会在 conda 环境内创建私有 `cat-cuda-home`/`cat-cuda-link` 软链接目录，避免最终链接阶段出现 `ld: cannot find -lcudart`。
- 安装 `ctc-align-cuda` 前会修正 `core.cu` 的 include，避免 nvcc 在 CUDA 12.x + PyTorch 2.x 组合下解析 pybind11 头文件时报 `expected template-name before '<' token`。
- `jiwer` 约束为 `>=2.2.0,<3.0`，以兼容 CAT 附带脚本使用的 `jiwer.compute_measures(...)` 老接口。
- `transformers` 约束为 `>=4.12.3,<4.54`，以兼容默认安装的 PyTorch 2.1.0。
- 如果旧环境曾安装到更高版本的 `transformers`，重跑脚本会在默认 profile 下重新应用该约束，避免 `torch.utils._pytree.register_pytree_node` 兼容性错误。
- 如需安装最新 PyTorch 和最新 `transformers`，必须显式设置 `LATEST_TORCH_TRANSFORMERS=1`；该模式不属于默认验证栈，安装后需要重新测试 `ctc-crf`、`warp-rnnt`、`warp-ctct`、`ctc-align-cuda` 等 native extensions。
- 安装 CAT 到 `~/CAT`。
- 写入 shell rc，使新终端能自动初始化 conda、设置 `CAT_ROOT` 和 `PATH`。

## 4. 常用参数

```bash
# 指定安装目录和环境名
CAT_INSTALL_ROOT=/data/CAT CONDA_ENV_NAME=cat310 bash install_cat_bundle.sh

# conda 不在 ~/miniconda3 时可显式指定
CONDA_HOME=/opt/miniconda3 bash install_cat_bundle.sh

# 如果 cat 环境存在但不完整，显式允许删除重建
RECREATE_CONDA_ENV=1 bash install_cat_bundle.sh

# 指定 CUDA/PyTorch 版本
CUDA_VERSION=12.1 TORCH_VERSION=2.1.0 TORCHAUDIO_VERSION=2.1.0 TORCHVISION_VERSION=0.16.0 bash install_cat_bundle.sh

# 显式选择最新 PyTorch + 最新 transformers，用于最新 LLM/Qwen 等场景。
# 该模式会重新编译 native extensions，安装后需要重新跑 ctc-crf 等 smoke tests。
LATEST_TORCH_TRANSFORMERS=1 CUDA_VERSION=12.4 bash install_cat_bundle.sh cat

# 如果只想覆盖 transformers 范围，也可以单独指定。
TRANSFORMERS_SPEC="transformers>=5" bash install_cat_bundle.sh cat

# 手工修复已有 cat 环境里的 transformers/torch 2.1 兼容问题
CONDA_HOME=/path/to/miniconda3 bash install_cat_bundle.sh cat
# 或直接在环境中执行：
conda run -n cat python -m pip install --upgrade "transformers>=4.12.3,<4.54"

# ctc-crf 会默认用 PyTorch 自动探测当前 GPU 架构；也可以显式指定。
# 这里的 80/89/90 是 GPU compute capability，不是 CUDA toolkit 版本。
CTC_CRF_CUDA_ARCH_LIST="80 90" bash install_cat_bundle.sh ctc-crf

# ctc-crf 会自动移除旧 setup.py 写死的 -std=c++14，
# 让 PyTorch BuildExtension 按当前版本管理 C++ extension 标准。

# 如果镜像没有 nvidia channel，默认会用官方 NVIDIA channel；也可以显式指定
NVIDIA_CONDA_CHANNEL=https://conda.anaconda.org/nvidia bash install_cat_bundle.sh

# 只使用 wheelhouse，不访问 pip 镜像
PIP_OFFLINE=1 bash install_cat_bundle.sh

# 目标机没有系统 nvcc 时，尝试从 conda 安装 CUDA 编译工具
INSTALL_CUDA_BUILD_TOOLKIT=1 bash install_cat_bundle.sh

# ctc-crf 编译时默认会安装并选择兼容 nvcc 的 GCC/G++。
# 如果需要禁用 conda 编译器安装，或手动指定 host compiler：
INSTALL_CONDA_BUILD_COMPILERS=0 CTC_CRF_CC=gcc-12 CTC_CRF_CXX=g++-12 bash install_cat_bundle.sh ctc-crf

# 如果 CUDA headers/runtime 不在常规位置，可显式指定。
# CUDA_CUDART_LIBRARY 可以指向版本化文件，例如 libcudart.so.12。
CUDA_INCLUDE_DIR=/usr/include CUDA_CUDART_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudart.so.12 bash install_cat_bundle.sh cat

# 只做 CAT Python 包的轻量安装时，可跳过本地 CUDA 源码依赖
INSTALL_LOCAL_SOURCE_DEPS=0 BUILD_CTC_CRF=0 bash install_cat_bundle.sh

# 默认保持原脚本语义，以 editable 方式安装本地第三方源码依赖。
# 如果 pip 25+ 因 legacy editable 行为变化导致失败，可切换为非 editable。
EDITABLE_LOCAL_SOURCE_DEPS=0 bash install_cat_bundle.sh cat

# 不写 shell 启动文件
WRITE_SHELL_RC=0 bash install_cat_bundle.sh

# 只构建 g2p 工具
bash install_cat_bundle.sh g2p-tool

# 只构建 fst-decoder，需要已有 Kaldi
KALDI_ROOT=/opt/kaldi bash install_cat_bundle.sh fst-decoder
```

## 5. Bundle 内容

```text
payload/source/CAT
payload/third_party/repos/ctcdecode
payload/third_party/repos/kenlm
payload/third_party/repos/Phonetisaurus
payload/third_party/repos/torch-gather
payload/third_party/repos/warp-rnnt
payload/third_party/repos/webdataset
payload/third_party/repos/warp-ctct
payload/third_party/repos/ctc-align-cuda
payload/third_party/tarballs/openfst-1.6.7.tar.gz
payload/third_party/tarballs/openfst-1.7.2.tar.gz
payload/third_party/tarballs/boost_1_67_0.tar.gz
payload/requirements/requirements.runtime.txt
```

目标机安装脚本不会运行 `git clone`。如果安装失败后重跑，它会复用已有 conda、环境和源码目录，但 native 编译步骤仍可能重新执行。
