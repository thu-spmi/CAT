#!/usr/bin/make -f
# Latent CTC Makefile (H100 Optimized Version - Fixed)

# 项目根目录
ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# H100 配置
CUDA_HOME ?= /usr/local/cuda-12.1
CUDA_VERSION := 12.1
H100_ARCH := 90
export PATH := $(CUDA_HOME)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_HOME)/lib64:$(LD_LIBRARY_PATH)
export TORCH_CUDA_ARCH_LIST := "$(H100_ARCH).0"

# 编译器设置
CC := gcc
CXX := g++
NVCC := $(CUDA_HOME)/bin/nvcc

# 并行编译
NPROC ?= $(shell nproc)

.PHONY: all GPUCTC CTCCRF clean distclean

all: CTCCRF

# === 1. GPU CTC 构建 (H100 优化) ===
GPUCTC:
	@echo "🔧 构建 GPU CTC (H100 优化)..."
	@mkdir -p gpu_ctc/build
	@cd gpu_ctc/build && cmake \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_COMPILER=$(NVCC) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_CUDA_ARCHITECTURES=$(H100_ARCH) \
		-DCUDA_TOOLKIT_ROOT_DIR=$(CUDA_HOME) \
		..
	@cd gpu_ctc/build && make -j $(NPROC)
	@echo "✅ GPU CTC 构建完成"

# === 2. Python 扩展构建 ===
CTCCRF: GPUCTC
	@echo "🔧 构建 Latent CTC Python 扩展..."
	@python3 setup.py install
	@echo "✅ Latent CTC 安装完成"

# === 3. 清理 ===
clean:
	@echo "🧹 清理构建文件..."
	@python3 setup.py clean --all >/dev/null 2>&1 || true
	@rm -rf gpu_ctc/build build/
	@rm -rf dist/ latent_ctc.egg-info/
	@echo "✅ 清理完成"

distclean: clean
	@echo "🔥 深度清理..."
	@echo "✅ 深度清理完成"

# === 4. 验证目标 ===
verify:
	@echo "✅ 验证 H100 环境..."
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "CUDA Version: $(shell $(NVCC) --version 2>/dev/null | grep release || echo 'not found')"
	@echo "GPU Architecture: $(H100_ARCH)"
	@nvidia-smi -q | grep "Product Name" | head -1 2>/dev/null || echo "GPU not found"
	@python3 -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name()}')" || echo "PyTorch not found"

# === 5. 调试目标 ===
debug:
	@echo "🐞 调试构建环境..."
	@cd gpu_ctc/build && make VERBOSE=1

# === 6. 重新构建 ===
rebuild: distclean all

# === 7. H100 专用构建 ===
h100: export TORCH_CUDA_ARCH_LIST := "9.0"
h100: CTCCRF
	@echo "🚀 H100 专用构建完成!"

# === 8. 帮助 ===
help:
	@echo "🚀 Latent CTC Makefile (H100 Optimized - Fixed Version)"
	@echo ""
	@echo "目标:"
	@echo "  all         - 构建所有组件 (默认)"
	@echo "  GPUCTC      - 仅构建 GPU CTC"
	@echo "  CTCCRF      - 仅构建 Python 扩展"
	@echo "  h100        - H100 专用构建"
	@echo "  verify      - 验证 H100 环境"
	@echo "  debug       - 调试构建"
	@echo "  clean       - 清理构建文件"
	@echo "  distclean   - 深度清理 (包括源码)"
	@echo "  rebuild     - 重新构建所有"
	@echo ""
	@echo "环境变量:"
	@echo "  CUDA_HOME   - CUDA 安装路径 (默认: /usr/local/cuda-12.1)"
	@echo "  NPROC       - 并行编译线程数 (默认: 自动检测)"
	@echo ""
	@echo "示例:"
	@echo "  make h100                   # H100 专用构建"
	@echo "  make -j8                    # 8 线程并行构建"
	@echo "  CUDA_HOME=/usr/local/cuda-12.0 make  # 指定 CUDA 版本"
