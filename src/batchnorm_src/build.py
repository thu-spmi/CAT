import glob
import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = []
headers = []
defines = []
with_cuda = False

abs_path = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(abs_path, 'internel/batchnormp_kernel.so')]

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['batchnorm_binding.c']
    headers += ['batchnorm_binding.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    'batchnorm_utils.pytorch',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects)

if __name__ == '__main__':
    ffi.build()
