'''
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang, Author: Hu Juntao (hujuntao_123@outlook.com)
          2019-2022 Tsinghua University, Author: Huahuan Zheng (maxwellzh@outlook.com)
Apache 2.0.
Build CTC-CRF pytorch binding
'''

import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


if not torch.cuda.is_available():
    raise Exception("CPU version is not implemented")


den_dir = os.path.realpath("./gpu_den/build")
ctc_dir = os.path.realpath("./gpu_ctc/build")

if __name__ == "__main__":
    setup(name='ctc_crf',
          version="0.1.1",
          packages=find_packages(),
          ext_modules=[
              CppExtension(
                  name='ctc_crf._C',
                  language='c++',
                  sources=['binding.cpp'],
                  library_dirs=[ctc_dir, den_dir],
                  libraries=['fst_den', 'warpctc'],
                  extra_link_args=['-Wl,-rpath,' +
                                   ctc_dir, '-Wl,-rpath,'+den_dir],
                  extra_compile_args=['-std=c++14',
                                      '-fPIC', '-I/usr/local/cuda/include']
              ),
          ],
          cmdclass={
              'build_ext': BuildExtension})
