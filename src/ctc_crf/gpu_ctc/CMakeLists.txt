cmake_minimum_required(VERSION 2.8)

project(ctc_release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -O2")
set(CMAKE_C_COMPILLER "usr/libexec/gcc")
include_directories(include)

FIND_PACKAGE(CUDA 6.5)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")

# need to be at least 30 or __shfl_down in reduce wont compile
IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
ENDIF()

IF (CUDA_VERSION GREATER 7.6)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF (CUDA_VERSION GREATER 8.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()


IF (CUDA_VERSION GREATER 10.0)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
set(CMAKE_SKIP_RPATH TRUE)

MESSAGE(STATUS "Building shared library with GPU support")
CUDA_ADD_LIBRARY(warpctc SHARED ctc_entrypoint.cu)
# IF (!Torch_FOUND)
#     TARGET_LINK_LIBRARIES(warpctc ${CUDA_curand_LIBRARY})
# ENDIF()
