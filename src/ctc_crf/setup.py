# Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
# Apache 2.0.
# Build CTC-CRF pytorch binding

import os
import torch
import platform
import sys
from torch.utils.ffi import create_extension

extra_compile_args = ['-std=c++11', '-fPIC']
headers = ['binding.h']
den_dir = os.path.realpath("./gpu_den/build")
ctc_dir = os.path.realpath("./gpu_ctc/build")

ffi = create_extension(
    name='ctc_crf_base',
    language='c++',
    with_cuda=True,
    headers=headers,
    sources=['binding.cpp'],
    library_dirs=[ctc_dir, den_dir],
    libraries=['fst_den', 'warpctc'],
    extra_link_args=['-Wl,-rpath,'+ ctc_dir, '-Wl,-rpath,'+den_dir],
    extra_compile_args=extra_compile_args)

if __name__ == "__main__":
    ffi.build()
