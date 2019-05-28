#!/bin/sh
set -e

# aarch64 is actually built natively and copied in...

SED=gsed

git clone --depth 1 -b v4.2.2 git://github.com/zeromq/libzmq.git libzmq
git clone --depth 1 -b v4.0.2 git://github.com/zeromq/czmq.git czmq

$SED -i 's/defined .HAVE_GETIFADDRS./0/' czmq/src/ziflist.c

LIBZMQ_ROOT=$HOME/one/phonelibs/zmq/libzmq

export ANDROID_NDK_ROOT=/opt/android-ndk

export ANDROID_BUILD_EXTRA_CFLAGS='-std=gnu99 -O2'
export ANDROID_BUILD_EXTRA_CXXFLAGS='-O2'

######## arm

export TOOLCHAIN_PATH=${ANDROID_NDK_ROOT}/toolchains/arm-linux-androideabi-4.9/prebuilt/darwin-x86_64/bin
export TOOLCHAIN_NAME=arm-linux-androideabi-4.9
export TOOLCHAIN_HOST=arm-linux-androideabi
export TOOLCHAIN_ARCH=arm
cd czmq/builds/android
./build.sh
cd ../../../
cp czmq/builds/android/prefix/arm-linux-androideabi-4.9/lib/libczmq.a \
   czmq/builds/android/prefix/arm-linux-androideabi-4.9/lib/libczmq.so \
   czmq/builds/android/prefix/arm-linux-androideabi-4.9/lib/libzmq.a \
   czmq/builds/android/prefix/arm-linux-androideabi-4.9/lib/libzmq.so ./arm/lib/
cp czmq/builds/android/prefix/arm-linux-androideabi-4.9/include/*.h ./arm/include/


######## aarch64

(cd libzmq && patch -p0 <../build_aarch64.patch)
(cd czmq && patch -p0 <../build_aarch64.patch)

# android-9 lacks aarch64.
$SED -i 's/android-9/android-24/' *zmq/builds/android/android_build_helper.sh
# For some reason gcc doesn't work for aarch64, but g++ does.
$SED -i 's/-lgnustl_shared/-l:libgnustl_static.a/' *zmq/builds/android/android_build_helper.sh

export TOOLCHAIN_PATH=${ANDROID_NDK_ROOT}/toolchains/aarch64-linux-android-4.9/prebuilt/darwin-x86_64/bin
export TOOLCHAIN_NAME=aarch64-linux-android-4.9
export TOOLCHAIN_HOST=aarch64-linux-android
export TOOLCHAIN_ARCH=arm64
cd czmq/builds/android
./build.sh
cd ../../../
cp czmq/builds/android/prefix/aarch64-linux-android-4.9/lib/libczmq.a \
   czmq/builds/android/prefix/aarch64-linux-android-4.9/lib/libczmq.so \
   czmq/builds/android/prefix/aarch64-linux-android-4.9/lib/libzmq.a \
   czmq/builds/android/prefix/aarch64-linux-android-4.9/lib/libzmq.so ./aarch64/lib/
cp czmq/builds/android/prefix/aarch64-linux-android-4.9/include/*.h ./aarch64/include/

# rm -rf czmq
echo SUCCESS
