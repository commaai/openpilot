#!/bin/sh
set -e
VERSION=4.0.2
LIBFILE=czmq-${VERSION}
wget https://github.com/zeromq/czmq/releases/download/v4.0.2/${LIBFILE}.tar.gz
tar -xzf ${LIBFILE}.tar.gz
INSTALL_PATH=$PWD
cd ${LIBFILE}
export CFLAGS=-I${INSTALL_PATH}/include
export LDFLAGS=-L${INSTALL_PATH}/lib
export PKG_CONFIG_PATH=-L${INSTALL_PATH}/lib/pkgconfig
./configure --prefix=${INSTALL_PATH}
make -j4 install
cd ../
rm -f bin/zmakecert
rm -rf ${LIBFILE}
rm -rf ${LIBFILE}.tar.gz
