#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

ARCHNAME=$(uname -m)
if [ -f /TICI ]; then
  ARCHNAME="larch64"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

if [ ! -d raylib_repo ]; then
  git clone https://github.com/raysan5/raylib.git raylib_repo
fi

cd raylib_repo
git fetch --tags origin 5.0
git checkout 5.0

git clean -xdff .
mkdir build
cd build
cmake ..
make -j$(nproc)

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

rm -rf $DIR/include
cp $DIR/raylib_repo/build/raylib/libraylib.a $INSTALL_DIR/
cp -r $DIR/raylib_repo/build/raylib/include $DIR
