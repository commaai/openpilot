#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

ARCHNAME="x86_64"
MAPLIBRE_FLAGS="-DMLN_QT_WITH_LOCATION=OFF"
if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
elif [[ "$OSTYPE" == "aarch64"* ]]; then
  ARCHNAME="aarch64"
  MAPLIBRE_FLAGS="$MAPLIBRE_FLAGS -DCMAKE_SYSTEM_NAME=Android -DANDROID_ABI=arm64-v8a"
fi

if [ ! -d maplibre_repo ]; then
  git clone git@github.com:maplibre/maplibre-native-qt.git $DIR/maplibre_repo
fi
cd $DIR/maplibre_repo
git fetch --all
git checkout 3726266e127c1f94ad64837c9dbe03d238255816
git submodule update --depth=1 --recursive --init

# build
mkdir -p build
cd build
cmake $MAPLIBRE_FLAGS $DIR/maplibre_repo
make -j$(nproc)

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

rm -rf $INSTALL_DIR/lib $DIR/include
mkdir -p $INSTALL_DIR/lib $INSTALL_DIR/include $DIR/include
cp -r $DIR/maplibre_repo/build/src/core/*.so* $INSTALL_DIR/lib
cp -r $DIR/maplibre_repo/build/src/core/include/* $INSTALL_DIR/include
cp -r $DIR/maplibre_repo/src/**/*.hpp $DIR/include

cd $DIR
rm -rf maplibre_repo
