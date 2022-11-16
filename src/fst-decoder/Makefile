# author: Huahuan Zheng (maxwellzh@outlook.com)
# This file is modified from kaldi/src/bin/Makefile

kaldi_root_src=${KALDI_ROOT}/src
EXTRA_CXXFLAGS = -Wno-sign-compare -I${kaldi_root_src}
include ${kaldi_root_src}/kaldi.mk

BINFILES = latgen-faster


OBJFILES =

ADDLIBS = ${kaldi_root_src}/decoder/kaldi-decoder.a ${kaldi_root_src}/lat/kaldi-lat.a ${kaldi_root_src}/lm/kaldi-lm.a \
          ${kaldi_root_src}/fstext/kaldi-fstext.a ${kaldi_root_src}/hmm/kaldi-hmm.a \
          ${kaldi_root_src}/transform/kaldi-transform.a ${kaldi_root_src}/gmm/kaldi-gmm.a \
          ${kaldi_root_src}/tree/kaldi-tree.a ${kaldi_root_src}/util/kaldi-util.a ${kaldi_root_src}/matrix/kaldi-matrix.a \
          ${kaldi_root_src}/base/kaldi-base.a


TESTFILES =

include ${kaldi_root_src}/makefiles/default_rules.mk
