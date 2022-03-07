# Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
#           2021      Tsinghua University, Author: Huahuan Zheng
# Apache 2.0.
# CTC-CRF Makefile

openfst_dir=${CURDIR}/openfst

.PHONY: OPENFST GPUCTC GPUDEN PATHWEIGHT CTCCRF
all: CTCCRF
OPENFST:
	if [ -f "openfst-1.6.7.tar.gz" ]; then rm openfst-1.6.7.tar.gz; fi
	wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz
	mkdir -p tmpopenfst
	tar -zxf openfst-1.6.7.tar.gz -C tmpopenfst
	cd tmpopenfst/openfst-1.6.7; ./configure --prefix=${openfst_dir}
	cd tmpopenfst/openfst-1.6.7; make -j $(nproc) && make install
	rm -rf tmpopenfst/ openfst-1.6.7.tar.gz
GPUCTC:
	mkdir -p gpu_ctc/build
	cd gpu_ctc/build && cmake ..
	cd gpu_ctc/build && make	
GPUDEN: OPENFST
	mkdir -p gpu_den/build
	cd gpu_den/build && cmake -D openfst_dir:STRING=${openfst_dir} ..
	cd gpu_den/build && make
PATHWEIGHT: OPENFST
	mkdir -p path_weight/build
	cd path_weight/build && cmake -D openfst_dir:STRING=${openfst_dir} ..
	cd path_weight/build && make
CTCCRF: GPUCTC GPUDEN PATHWEIGHT
	python3 setup.py install

clean:
	rm -rf gpu_{ctc,den}/build path_weight/build/ build/ \
		openfst/ tmpopenfst/ *.tar.gz \
		dist/ ctc_crf.egg-info/
	
