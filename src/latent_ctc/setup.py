#!/usr/bin/env python3
"""
Latent CTC Python 扩展构建脚本 (H100 Optimized - Fixed for PyTorch 2.x)
"""

import os
import sys
import torch
import subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths

def check_cuda_compatibility():
    """检查 CUDA 兼容性"""
    print("🔍 检查 CUDA 兼容性...")
    
    # 检查 PyTorch CUDA 版本
    torch_cuda_version = torch.version.cuda
    print(f"PyTorch CUDA 版本: {torch_cuda_version}")
    
    # 检查系统 CUDA 版本
    try:
        nvcc_version = subprocess.check_output(['nvcc', '--version'], 
                                              stderr=subprocess.DEVNULL,
                                              universal_newlines=True)
        cuda_version_line = [line for line in nvcc_version.split('\n') if 'release' in line][0]
        system_cuda_version = cuda_version_line.split('release ')[1].split(',')[0]
        print(f"系统 CUDA 版本: {system_cuda_version}")
    except Exception as e:
        print(f"⚠️  无法获取系统 CUDA 版本: {e}")
        print(f"💡 使用 PyTorch 版本: {torch_cuda_version}")
        system_cuda_version = torch_cuda_version
    
    # 检查 GPU 架构
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_cap = torch.cuda.get_device_capability()
        print(f"GPU 型号: {gpu_name}")
        print(f"GPU 架构: {gpu_cap}")
        
        if gpu_cap[0] < 9:
            print(f"⚠️  检测到非 H100 GPU (架构 {gpu_cap[0]}.{gpu_cap[1]})")
            print(f"💡 建议: 为获得最佳性能，请使用 H100 GPU (架构 9.0)")
    else:
        print("❌ 没有 CUDA GPU 可用!")
        sys.exit(1)
    
    return system_cuda_version

def get_build_paths():
    """获取构建路径"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # GPU CTC 构建路径
    gpu_ctc_dir = os.path.join(root_dir, 'gpu_ctc', 'build')
    if not os.path.exists(gpu_ctc_dir):
        print(f"❌ GPU CTC 构建目录不存在: {gpu_ctc_dir}")
        print(f"💡 请先运行: make GPUCTC")
        sys.exit(1)
    
    return {
        'root_dir': root_dir,
        'gpu_ctc_dir': gpu_ctc_dir,
    }


def get_pytorch_include_paths():
    """获取 PyTorch include 路径 (兼容新旧版本)"""
    try:
        # 新版本 PyTorch (推荐方式)
        includes = include_paths(cuda=True)
        print(f"✅ 使用新方式获取 PyTorch include 路径: {includes}")
        return includes
    except Exception as e:
        print(f"⚠️  新方式失败: {e}")
    
    try:
        # 旧版本兼容方式
        torch_include = os.path.join(os.path.dirname(torch.__file__), 'include')
        includes = [torch_include]
        if os.path.exists(os.path.join(torch_include, 'ATen')):
            includes.append(os.path.join(torch_include, 'ATen'))
        if os.path.exists(os.path.join(torch_include, 'torch', 'csrc', 'api', 'include')):
            includes.append(os.path.join(torch_include, 'torch', 'csrc', 'api', 'include'))
        
        print(f"✅ 使用旧方式获取 PyTorch include 路径: {includes}")
        return includes
    except Exception as e:
        print(f"⚠️  旧方式也失败: {e}")
        print("❌ 无法获取 PyTorch include 路径!")
        sys.exit(1)

def get_cuda_include_paths():
    """获取 CUDA include 路径"""
    cuda_includes = []
    
    # 尝试从环境变量获取
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if os.path.exists(os.path.join(cuda_home, 'include')):
        cuda_includes.append(os.path.join(cuda_home, 'include'))
    
    # 尝试从系统路径获取
    system_cuda_includes = [
        '/usr/local/cuda/include',
        '/usr/include',
        '/opt/cuda/include'
    ]
    
    for path in system_cuda_includes:
        if os.path.exists(path) and path not in cuda_includes:
            cuda_includes.append(path)
    
    # 验证 CUDA 头文件
    cuda_headers = ['cuda.h', 'cuda_runtime.h', 'curand.h']
    found_headers = []
    
    for include_path in cuda_includes:
        for header in cuda_headers:
            if os.path.exists(os.path.join(include_path, header)):
                found_headers.append(header)
    
    if len(found_headers) == len(cuda_headers):
        print(f"✅ 找到所有 CUDA 头文件: {found_headers}")
    else:
        print(f"⚠️  部分 CUDA 头文件缺失，找到: {found_headers}")
        print(f"💡 可能需要安装 CUDA toolkit 或设置 CUDA_HOME")
    
    return cuda_includes

if __name__ == "__main__":
    # 检查 CUDA 兼容性
    cuda_version = check_cuda_compatibility()
    
    # 获取构建路径
    paths = get_build_paths()
    
    print(f"✅ 构建路径验证通过:")
    print(f"   GPU CTC: {paths['gpu_ctc_dir']}")
    
    # 环境变量设置
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # H100 架构
    
    # 获取 PyTorch 和 CUDA include 路径
    pytorch_includes = get_pytorch_include_paths()
    cuda_includes = get_cuda_include_paths()
    
    # 编译器标志
    extra_compile_args = [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '-fPIC',
        '-fopenmp',
        '-march=native',
        '-mtune=native',
        '-Wno-unused-parameter',
        '-Wno-missing-field-initializers',
        '-Wno-deprecated-declarations'
    ]
    
    # CUDA 标志
    cuda_flags = [
        '-O3',
        '-DNDEBUG',
        '-lineinfo',
        '-use_fast_math',
        '-Xcompiler=-fPIC,-fopenmp,-march=native,-mtune=native',
        '--std=c++17',
        '-gencode=arch=compute_90,code=sm_90',
        '-gencode=arch=compute_90,code=compute_90',
        '-Xptxas=-v'
    ]
    
    # 链接参数
    extra_link_args = [
        '-Wl,-rpath,' + paths['gpu_ctc_dir'],
        '-fopenmp',
        '-lgomp',
        '-lcudart',
        '-lcurand'
    ]
    
    # 库目录
    library_dirs = [
        paths['gpu_ctc_dir'],
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ]
    
    # 库文件
    libraries = [
        'warpctc',      # GPU CTC
    ]
    
    # 头文件目录
    include_dirs = [
        *pytorch_includes,
        *cuda_includes,
        '/usr/include',
        '/usr/local/include'
    ]
    
    # 源文件
    sources = ['binding.cpp']
    
    print("🔧 配置 Latent CTC 扩展...")
    print(f"包含目录: {include_dirs}")
    print(f"库目录: {library_dirs}")
    print(f"编译器标志: {extra_compile_args}")
    print(f"CUDA 标志: {cuda_flags}")
    
    # 验证关键文件存在
    print("\n🔍 验证关键文件存在...")
    required_files = [
        os.path.join(paths['gpu_ctc_dir'], 'libwarpctc.so'),
        'binding.cpp'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            if 'binding.cpp' in file_path:
                print("💡 binding.cpp 应该在当前目录，请确认文件位置")
            sys.exit(1)
    
    # 动态获取 PyTorch 库路径
    try:
        torch_lib_path = library_paths(cuda=True)[0]
        if torch_lib_path not in library_dirs:
            library_dirs.insert(0, torch_lib_path)
        print(f"✅ 添加 PyTorch 库路径: {torch_lib_path}")
    except Exception as e:
        print(f"⚠️  无法获取 PyTorch 库路径: {e}")
    
    setup(
        name='latent_ctc',
        version="1.0.2",  # 更新版本号
        description='Latent CTC for H100 GPUs (Fixed for PyTorch 2.x)',
        author='saier',
        packages=find_packages(),
        ext_modules=[
            CUDAExtension(
                name='latent_ctc._C',
                sources=sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args={
                    'cxx': extra_compile_args,
                    'nvcc': cuda_flags
                },
                extra_link_args=extra_link_args,
                define_macros=[('WITH_CUDA', None)]
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=True)
        },
        install_requires=[
            'torch>=1.12.0',
            'numpy>=1.20.0'
        ],
        python_requires='>=3.8',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ]
    )
    
    print("🎉 Latent CTC 扩展构建完成!")