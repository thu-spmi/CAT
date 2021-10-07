# Installation

[中文Chinese](install_ch.md)

* [Install dependencies](#dependencies)
  * [PyTorch](#pytorch)
  * [Kaldi](#kaldi)
* [Install CAT](#cat)

This is a **step-by-step** tutorial of installation of CAT, including the installation of dependencies `PyTorch` and `Kaldi`.

## Dependencies<a id="dependencies"></a>

CAT requires two (main) dependencies: `PyTorch` and `Kaldi`. Some other dependent packages such as `h5py` for reading data is not included here, which can be easily installed with `pip/pip3`. 

We recommend to install CAT and its dependencies in `conda` environment, otherwise you may require root permission for installing some libraries.

And note that CAT is relied on CUDA availability. Directly running with CPU is not supported yet.

### PyTorch <a id="pytorch"></a>

1. Locate your CUDA package.

   ```bash
   whereis cuda
   ```

   The output includes the path to your CUDA, normally `/usr/local/cuda`, then check the  version.

   ```bash
   <path to cuda>/bin/nvcc -V
   ```

2. Install the corresponding PyTorch version.

   The [official guide](https://pytorch.org/get-started/locally/) of PyTorch gives very details of installing the latest stable version with different methods. 

   For previous version of PyTorch or if your CUDA version not listed, refer to [this](https://pytorch.org/get-started/previous-versions/) and find your suitable version. 

   Python3 with PyTorch 1.1+ are supported by CAT. For previous Python and PyTorch support, please refer to CAT v1 branch.
   
3. Check the CUDA environment used to build PyTorch.

   ```
   python3 -m torch.utils.collect_env
   # output would be something like this:
   # 
   # Collecting environment information...
   # PyTorch version: 1.8.1
   # Is debug build: False
   # CUDA used to build PyTorch: 10.2
   # ...
   ```

   

### Kaldi<a id="kaldi"></a>

Kaldi official installation guide

```bash
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi && cat INSTALL
```

Note that IRSTLM/SRILM/Kaldi LM tools may be required in some recipes. 

The directory of kaldi is needed in following steps, denoted as `<path to kaldi>`.

## CAT<a id="cat"></a>

1. Get source codes from GitHub. Denote the path as `<path to CAT>`.

   ```bash
   git clone https://github.com/thu-spmi/CAT
   export PATH_CAT=$(readlink -f <path to CAT>)
   export PATH_Kaldi=$(readlink -f <path to kaldi>)
   ```

2. Install python packages

   ```bash
   cd $PATH_CAT
   python -m pip install --user -r requirements.txt
   ```

3. Install the kaldi-patch of CAT into kaldi.

   ```bash
   # copy the patch file into kaldi
   cp $PATH_CAT/src/kaldi-patch/latgen-faster.cc $PATH_Kaldi/src/bin/
   # change directory
   cd $PATH_Kaldi/src/bin/
   ```

   Edit the `Makefile`:

   Add a string  `"latgen-faster"` to the `BINFILES` list. Then compile.

   ```bash
   make
   ```

   If everything goes well, there should be  a new file `kaldi/src/bin/latgen-faster`.

4. Install CTC-CRF module.

   ```bash
   # change directory
   cd $PATH_CAT/src/ctc_crf/
   ```

   CTC-CRF module will be installed as a python module. So, before going to next step, please ensure the python in your current path is the one you are to use with CAT:

   ```bash
   which python
   # or
   which python3
   ```
   
   Compile and install the package.
   
   ```bash
   # run as root if the command raise a "permission denied" error.
   # gcc-5 & g++-5 is also OK if there is no gcc-6 in your machine.
   CC=gcc-6 CXX=g++-6 make
   ```


   If encounter any issue, clean up previous buildings via

   ```bash
   make clean
   ```
   
5. Try import CTC-CRF module.

   ```bash
   python -c "import ctc_crf"
   ```

   There should be no error message thrown out.

6. A few further things.

   Change the `KALDI_ROOT=...` in `CAT/egs/<task>/path.sh` to `<path to kaldi>`. Here the `<task>` is your ASR task name, like `wsj` and `swbd`. 

   And, in `CAT/egs/wsj`, set the correct links

   ```bash
   cd $PATH_CAT/egs/wsj
   ln -snf $PATH_Kaldi/egs/wsj/s5/steps steps
   ln -snf $PATH_Kaldi/egs/wsj/s5/utils utils
   ```

6. Enjoy it! :rocket:
