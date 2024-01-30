#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

ARCHNAME="x86_64"
if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

if [ ! -d maplibre_repo ]; then
  git clone git@github.com:maplibre/maplibre-native-qt.git $DIR/maplibre_repo
fi
cd maplibre_repo
git fetch --all
# git checkout (on main right now)
git submodule update --depth=1 --recursive --init

#build
mkdir -p build && cd build
cmake -DMLN_QT_WITH_LOCATION=OFF $DIR/maplibre_repo
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
