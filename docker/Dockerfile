FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

# Install Kaldi
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake wget flac build-essential git subversion automake \
    autoconf unzip sox gfortran libtool python2.7 dos2unix
RUN ln -sf /bin/bash /bin/sh

RUN pip install mkl-include h5py tensorboard && \
    git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi/tools && \
    make -j $(nproc) && \
    ./extras/install_kaldi_lm.sh && \
    cd /opt/kaldi/src && \
    ./configure --debug-level=0 --shared --use-cuda --mkl-root=/opt/conda --mkl-libdir=/opt/conda/lib && \
    sed 's:-O1:-O2:g' kaldi.mk \
    make depend -j $(nproc) && \
    make -j $(nproc)

COPY . /opt/CAT

# Step 1. Copy src/kaldi-patch/latgen-faster.cc to kaldi src/bin, and compile.
COPY src/kaldi-patch/latgen-faster.cc /opt/kaldi/src/bin
WORKDIR /opt/kaldi/src/bin
RUN sed -i "s:BINFILES =:BINFILES = latgen-faster:g" Makefile
RUN make latgen-faster

# Step 2. cd src/ctc_crf, and run the commands below:
# make OPENFST=/path/to/your/openfst
# For pytorch version 1.0+, use python setup_1_0.py install in the ctc_crf/Makefile.
WORKDIR /opt/CAT/src/ctc_crf
RUN OPENFST=/opt/kaldi/tools/openfst make GPUCTC GPUDEN PATHWEIGHT
RUN python setup_1_0.py install

# Cleanup
RUN apt-get clean && rm /opt/kaldi/src/*/*.o

WORKDIR /workspace
