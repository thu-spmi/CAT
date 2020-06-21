#!/usr/bin/env python3
'''
Copyright 2018-2019 Tsinghua University, Author: Hu Juntao (hujuntao_123@outlook.com)
Apache 2.0.
This script is used to install ctc_crf_base_1_0 which depends on the ctc_crf native codes.
In this script we use cpp codes binding_1_0.h, binding_1_0.cpp to integrate the ctc fuctions.
This install script is used for the pytorch version 1.0.0 or later.
'''

import os
import torch
import platform
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

extra_compile_args = ['-std=c++14', '-fPIC','-I/usr/local/cuda/include']
headers = ['binding_1_0.h']
den_dir = os.path.realpath("./gpu_den/build")
ctc_dir = os.path.realpath("./gpu_ctc/build")

if __name__ == "__main__":
    setup(name='ctc_crf_base',
        ext_modules=[
            CppExtension(
              name='ctc_crf_base',
              language='c++',
              with_cuda=True,
              headers=headers,
              sources=['binding_1_0.cpp'],
              library_dirs=[ctc_dir, den_dir],
              libraries=['fst_den', 'warpctc'],
              extra_link_args=['-Wl,-rpath,'+ ctc_dir, '-Wl,-rpath,'+den_dir],
              extra_compile_args=extra_compile_args
              ),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
        )
