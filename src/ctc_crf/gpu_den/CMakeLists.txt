cmake_minimum_required(VERSION 2.8)
project(crf_den)

set(openfst_dir "fstdir" CACHE STRING "fst directory")
message(${openfst_dir})

include_directories(${openfst_dir}/include)
link_libraries(${openfst_dir}/lib/libfst.so)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 --std=c++11")

FIND_PACKAGE(CUDA 6.5)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")

# need to be at least 30 or __shfl_down in reduce wont compile
IF (CUDA_VERSION VERSION_LESS "11.0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
ENDIF()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

IF (CUDA_VERSION VERSION_GREATER "7.6")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "11.0") OR (CUDA_VERSION VERSION_EQUAL "11.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
ENDIF()

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

add_library(fst_read SHARED fst_read.cc)
cuda_add_library(fst_den SHARED den_calculate.cu)
target_link_libraries(fst_den fst_read)
