# Installation of CAT

This is a **step-by-step** tutorial of installation of CAT, especially for beginners, including the installation of dependencies `PyTorch` and `Kaldi`.

## Dependencies

CAT requires two (main) dependencies: `PyTorch` and `Kaldi`. Some other dependent packages such as `h5py` for reading data is not included here, which can be easily installed with `pip/pip3`. 

We recommend to install CAT and its dependencies in `conda` environment, otherwise you may require root permission for installing some libraries.

And note that CAT is relied on CUDA availability. Directly running with CPU is not supported yet.

### PyTorch 

1. Check your CUDA version.

   ```shell
   $ whereis cuda
   ```

   The output includes the path to your CUDA, normally `/usr/local/cuda`, then check the  version file.

   ```shell
   $ cat <path to CUDA>/version.txt
   ```

2. Install the corresponding PyTorch version.

   The [official guide](https://pytorch.org/get-started/locally/) of PyTorch gives very details of installing the latest stable version with different methods. 

   For previous version of PyTorch or if your CUDA version not listed, refer to [this](https://pytorch.org/get-started/previous-versions/) and find your suitable version. 

   Python2 with PyTorch 0.4+ or Python3 with PyTorch 1.0+ are supported by CAT.

### Kaldi

Please refer to this step-by-step installation guide: http://jrmeyer.github.io/asr/2016/01/26/Installing-Kaldi.html

Note that IRSTLM tool is required. The directory of kaldi is needed in following steps, denoted as `<path to kaldi>`.

### OpenFST

1. Create a temporary directory and change the working directory to it.

   ```shell
   $ mkdir tmpfst && cd tmpfst
   ```

2. Get OpenFST-1.6.7 file and extract.

   ```shell
   # fetch the file from source
   $ wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz
   # extract
   $ tar -xf openfst-1.6.7.tar.gz
   # change directory to extracted one
   $ cd openfst-1.6.7/
   ```

3. Make configuration of compiling. Without `--prefix` option, the OpenFST will  be installed to `/usr/local`. Individual directory for current user is recommended.

   ```shell
   $ ./configure --prefix=<path to openfst>
   ```

4. Make and install. This step may require root permission up to the `<path to openfst>` you specified in previous step.

   ```shell
   make && make install
   ```

5. [OPTIONAL] Remove the temporary directory.

   ```shell
   $ cd ../..
   $ rm -r tmpfst/
   ```

## Install CAT

1. Get source codes from GitHub. Denote the path as `<path to CAT>`.

   ```shell
   $ git clone https://github.com/thu-spmi/CAT
   ```

2. Install the kaldi-patch of CAT into kaldi.

   ```shell
   # copy the patch file into kaldi
   $ cp <path to CAT>/src/kaldi-patch/latgen-faster.cc <path to kaldi>/src/bin/
   # change directory
   $ cd <path to kaldi>/src/bin/
   ```

   Edit the `Makefile`:

   Add a string  `"latgen-faster"` to the `BINFILES` list. Then compile.

   ```shell
   $ make
   ```

   If everything goes well, there should be  two new files in `<path to kaldi>/src/bin/`: `latgen-faster` and `latgen-faster.o`.

3. Install CTC-CRF module.

   ```shell
   # change directory
   $ cd <path to CAT>/src/ctc_crf/
   ```

   CTC-CRF module will be installed as a python module. So, before going to next step, please ensure the python (or python3) in your current path is the one you are to use with CAT:

   ```shell
   # for python2
   $ which python
   # for python3
   $ which python3
   ```

   For Python2 installation, run following command and go to step 4.

   ```shell
   # run as root if the command raise a permission denied error.
   $ make OPENFST=<path to openfst>
   ```

   For Python3, firstly open the `Makefile` and comment the last line of command. Like this

   ```makefile
   ...
   CTCCRF: GPUCTC GPUDEN PATHWEIGHT
   #	python setup.py
   ```

   Then compile dependencies of CTC-CRF module.

   ```shell
   $ make OPENFST=<path to openfst>
   ```

   Finally, install the CTC-CRF module.

   ```shell
   # run as root if the command raise a permission denied error.
   $ python3 setup_1_0.py install
   ```

4. Try import CTC-CRF module.

   ```shell
   $ python
   # or python3
   $ python3
   >>> import torch
   >>> import ctc_crf_base
   ```

   There should be no error message thrown out. And note that because CTC-CRF module is relied on torch, you should import `torch` before import `ctc_crf_base`.

5. Further tiny works.

   Change the `KALDI_ROOT=...` in `<path to CAT>/scripts/ctc-crf/kaldi_io.py` and `<path to CAT>/egs/<task>/path.sh` to `<path to kaldi>`. Here the `<task>` is your ASR task directory, like `wsj` and `swbd`.

6. Enjoy it! :rocket: