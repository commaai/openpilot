#!/usr/bin/env bash
set -e
rm -rf nng-1.1.1
rm -rf x64
wget https://github.com/nanomsg/nanomsg/archive/1.1.5.tar.gz -O nanomsg-1.1.5.tar.gz
tar xvf nanomsg-1.1.5.tar.gz
pushd nanomsg-1.1.5

mkdir build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/home/batman/one/phonelibs/nanomsg/x64 ..
make -j
make install

popd
popd

rm nanomsg-1.1.5.tar.gz
rm -r nanomsg-1.1.5
