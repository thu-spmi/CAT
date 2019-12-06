PYTORCH_LIB_DIR := /home/ouzj02/xianghy/pytorch0.41/torch/lib


PYTHON := python
NVCC_COMPILE := nvcc -c --std=c++11 -o 
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared

# File structure.
INCLUDE_DIRS := include/TH include/THC include/THNN include/THCUNN include .
TORCH_FFI_BUILD := build.py
BN_KERNEL := internel/batchnormp_kernel.so
TORCH_FFI_TARGET := pytorch/_batch_norm.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$(PYTORCH_LIB_DIR)/$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(BN_KERNEL) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BN_KERNEL):batchnorms_cuda_kernel.cu
	@mkdir -p internel
	@touch __init__.py
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS) -Isrc

clean:
	$(RM_RF) $(BUILD_DIR)
