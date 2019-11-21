# Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
# Apache 2.0.
# CTC-CRF Makefile

openfst_dir := /opt/scouce/kaldi/tools/openfst-1.6.2

.PHONY: GPUCTC GPUDEN CTCCRF clean
all: CTCCRF
GPUCTC:
	mkdir -p gpu_ctc/build
	cd gpu_ctc/build && cmake ..
	cd gpu_ctc/build && make	
GPUDEN:
	mkdir -p gpu_den/build
	cd gpu_den/build && cmake -Dopenfst_dir:STRING=${openfst_dir} ..
	cd gpu_den/build && make	
CTCCRF: GPUCTC GPUDEN
	python setup.py
clean:
	rm -rf gpu_ctc/build
	rm -rf gpu_den/build
	rm -rf ctc_crf_base
