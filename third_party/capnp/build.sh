#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

ARCHNAME=$(uname -m)
if [ -f /TICI ]; then
  ARCHNAME="larch64"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

CAPNP_VERSION="v1.0.1"

cd $DIR
if [ ! -d capnproto ]; then
  git clone --single-branch --depth 1 -b $CAPNP_VERSION https://github.com/capnproto/capnproto.git
fi

cd capnproto
git checkout $CAPNP_VERSION

# build
mkdir -p build
cd build

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DWITH_OPENSSL=OFF"

if [[ "$OSTYPE" == "darwin"* ]]; then
  CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_OSX_ARCHITECTURES=arm64;x86_64"
fi

cmake $CMAKE_FLAGS ..
make -j$(nproc)

# install
INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR/lib $INSTALL_DIR/bin

rm -rf $DIR/include
mkdir -p $DIR/include

# only copy the headers we need (capnp and kj)
cp -r $DIR/capnproto/c++/src/capnp $DIR/include/
cp -r $DIR/capnproto/c++/src/kj $DIR/include/

# remove test, compat, and other unnecessary headers
find $DIR/include -name "*-test*" -delete
find $DIR/include -name "*_test*" -delete
find $DIR/include -name "*.c++" -delete
find $DIR/include -name "*.capnp.h" -delete
find $DIR/include -name "*.capnp" ! -name "c++.capnp" -delete
rm -rf $DIR/include/capnp/compat
rm -rf $DIR/include/capnp/compiler
rm -rf $DIR/include/capnp/testdata
rm -rf $DIR/include/kj/compat
find $DIR/include -name "CMakeLists.txt" -o -name "BUILD.bazel" -o -name "*.bzl" \
  -o -name "*.ekam-rule" -o -name "*.ekam-manifest" | xargs rm -f

# runtime libraries (linked by the project)
cp $DIR/capnproto/build/c++/src/capnp/libcapnp.a $INSTALL_DIR/lib/
cp $DIR/capnproto/build/c++/src/kj/libkj.a $INSTALL_DIR/lib/

# compiler libraries and tools (needed to compile .capnp schemas)
cp $DIR/capnproto/build/c++/src/capnp/libcapnpc.a $INSTALL_DIR/lib/
cp $DIR/capnproto/build/c++/src/capnp/libcapnp-json.a $INSTALL_DIR/lib/
cp $DIR/capnproto/build/c++/src/capnp/capnpc-c++ $INSTALL_DIR/bin/
cp $DIR/capnproto/build/c++/src/capnp/capnp $INSTALL_DIR/bin/
# capnpc is a symlink to capnp
ln -sf capnp $INSTALL_DIR/bin/capnpc

# strip binaries
strip $INSTALL_DIR/bin/capnp $INSTALL_DIR/bin/capnpc-c++

## To create universal binary on Darwin:
## ```
## for f in libcapnp.a libkj.a libcapnpc.a libcapnp-json.a; do
##   lipo -create -output Darwin/lib/$f path-to-x64/lib/$f path-to-arm64/lib/$f
## done
## for f in capnp capnpc-c++; do
##   lipo -create -output Darwin/bin/$f path-to-x64/bin/$f path-to-arm64/bin/$f
## done
## ```
