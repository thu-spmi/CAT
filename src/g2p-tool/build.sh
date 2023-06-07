#!/bin/bash
set -e -u

# build and install openfst-1.7.2
# openfst lib would be installed at openfst-1.7.2/
export OFST_PATH="$(readlink -f openfst-1.7.2)"
[ ! -f $OFST_PATH/.done ] && {
  wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz
  tar -zxf openfst-1.7.2.tar.gz
  mv openfst-1.7.2 openfst-1.7.2-build
  cd openfst-1.7.2-build

  ./configure --prefix=$OFST_PATH \
    --enable-static --enable-shared \
    --enable-far --enable-ngram-fsts

  make -j && make install
  cd - >/dev/null
  rm -rf openfst-1.7.2-build openfst-1.7.2.tar.gz
  touch $OFST_PATH/.done
}

# install Phonetisaurus
[ ! -d Phonetisaurus ] &&
  git clone https://github.com/AdolfVonKleist/Phonetisaurus.git

cd Phonetisaurus
[ ! -f build/.done ] && {
  python -m pip install pybindgen
  PYTHON=python ./configure --enable-python \
    --with-openfst-includes=${OFST_PATH}/include \
    --with-openfst-libs=${OFST_PATH}/lib \
    --prefix=$(readlink -f build)

  make && make install
  touch build/.done
}

cd python
cp ../.libs/Phonetisaurus.so ./
python setup.py install
cd ../../../bin/
ln -snf ../g2p-tool/Phonetisaurus/build/bin/* ./

echo "$0 done" && exit 0
