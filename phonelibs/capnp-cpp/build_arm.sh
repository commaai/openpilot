#!/bin/bash
set -e

NDK=/opt/android-ndk
SYSROOT=/opt/android-ndk/platforms/android-23/arch-arm

LLVM_PATH=/opt/android-ndk/toolchains/llvm/prebuilt/darwin-x86_64/bin
TOOLS_PATH=/opt/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/arm-linux-androideabi/bin

export CPP=/opt/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/bin/arm-linux-androideabi-cpp
export AR=${TOOLS_PATH}/ar
export AS=${TOOLS_PATH}/as
export NM=${TOOLS_PATH}/nm
export CC=${LLVM_PATH}/clang
export CXX="${LLVM_PATH}/clang++ -target armv7-none-linux-androideabi -gcc-toolchain /opt/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64"
export LD=${TOOLS_PATH}/ld
export RANLIB=${TOOLS_PATH}/ranlib
export SED=gsed

export CPPFLAGS="--sysroot=${SYSROOT} -I${SYSROOT}/usr/include"

export CFLAGS="-target armv7-none-linux-androideabi \
  -isystem ${SYSROOT}/usr/include \
  --sysroot=${SYSROOT} \
  -I${SYSROOT}/usr/include \
  -gcc-toolchain /opt/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64 \
  -no-canonical-prefixes -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16"

export CXXFLAGS="$CFLAGS -std=c++11 -stdlib=libstdc++ \
  -I/opt/android-ndk/sources/cxx-stl/gnu-libstdc++/4.9/include \
  -I/opt/android-ndk/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include \
  -I/opt/android-ndk/sources/cxx-stl/gnu-libstdc++/4.9/include/backward"

export LDFLAGS="-target armv7-none-linux-androideabi \
  -gcc-toolchain /opt/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64 \
  -L${SYSROOT}/usr/lib"

# /opt/android-ndk/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/libgnustl_static.a

./configure --host=arm-linux-androideabi --disable-shared --with-external-capnp
make -j4
# itll fail when it gets to libtool stuff...
