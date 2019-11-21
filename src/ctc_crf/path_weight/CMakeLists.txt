cmake_minimum_required(VERSION 2.8)
project(crf_den)

#include_directories(/home/ouzj02/kaldi-master/tools/openfst-1.6.7/include)
#link_libraries(/home/ouzj02/kaldi-master/tools/openfst-1.6.7/lib/libfst.so)

include_directories(/home/ouzj02/ankeyu/kaldi-5.3/tools/openfst-1.6.5/include)
link_libraries(/home/ouzj02/ankeyu/kaldi-5.3/tools/openfst-1.6.5/lib/libfst.so)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 --std=c++11")

add_executable(path_weight path_weight.cc)
