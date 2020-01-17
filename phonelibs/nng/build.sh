#!/usr/bin/env bash
set -e
rm -rf nng-1.1.1
rm -rf x64

wget https://codeload.github.com/nanomsg/nng/tar.gz/v1.1.1 -O nng-1.1.1.tar.gz
tar xvf nng-1.1.1.tar.gz
pushd nng-1.1.1

mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/batman/one/phonelibs/nng/x64 ..
make -j
make install

popd
popd

rm -r nng-1.1.1
rm nng-1.1.1.tar.gz
