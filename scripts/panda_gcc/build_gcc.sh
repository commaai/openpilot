#!/usr/bin/env bash
# https://blog.tan-ce.com/gcc-bare-metal/
# https://imvoid.wordpress.com/2013/05/01/building-the-gnu-arm-toolchain-for-bare-metal/
set -e

BINUTILS=binutils-2.32
GCC=gcc-4.7.1

mkdir -p src
pushd src
if [ ! -d $BINUTILS ]; then
  wget ftp://ftp.gnu.org/gnu/binutils/$BINUTILS.tar.bz2
  tar -xf $BINUTILS.tar.bz2
fi
popd

# TODO: replace with /usr
mkdir -p out
PREFIX=$PWD/out

mkdir -p build/$BINUTILS
pushd build/$BINUTILS
../../src/$BINUTILS/configure --target=arm-none-eabi \
  --build=aarch64-unknown-linux-gnu \
  --prefix=$PREFIX --with-cpu=cortex-m4 \
  --with-mode=thumb \
  --disable-nls \
  --disable-werror
make -j4 all
make install
popd

mkdir -p src
pushd src
if [ ! -d $GCC ]; then
  wget ftp://ftp.gnu.org/gnu/gcc/$GCC/$GCC.tar.bz2
  tar -xf $GCC.tar.bz2

  cd $GCC
  contrib/download_prerequisites
fi

popd

export PATH="$PREFIX/bin:$PATH"

mkdir -p build/$GCC
pushd build/$GCC
../../src/$GCC/configure --target=arm-none-eabi \
  --build=aarch64-unknown-linux-gnu \
  --disable-libssp --disable-gomp --disable-libstcxx-pch --enable-threads \
  --disable-shared --disable-libmudflap \
  --prefix=$PREFIX --with-cpu=cortex-m4 \
  --with-mode=thumb --disable-multilib \
  --enable-interwork \
  --enable-languages="c" \
  --disable-nls \
  --disable-libgcc
make -j4 all-gcc
make install-gcc
popd

# replace stdint.h with stdint-gcc.h for Android compatibility
mv $PREFIX/lib/gcc/arm-none-eabi/4.7.1/include/stdint-gcc.h $PREFIX/lib/gcc/arm-none-eabi/4.7.1/include/stdint.h


