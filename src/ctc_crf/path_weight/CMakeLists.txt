cmake_minimum_required(VERSION 2.8)
project(crf_den)

set(openfst_dir "fstdir" CACHE STRING "fst directory")
message(${openfst_dir})

include_directories(${openfst_dir}/include)
link_libraries(${openfst_dir}/lib/libfst.so)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 --std=c++11")

add_executable(path_weight path_weight.cc)
