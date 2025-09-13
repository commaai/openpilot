#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

RAYLIB_PLATFORM="PLATFORM_DESKTOP"

ARCHNAME=$(uname -m)
if [ -f /TICI ]; then
  ARCHNAME="larch64"
  RAYLIB_PLATFORM="PLATFORM_COMMA"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

INSTALL_H_DIR="$DIR/include"
rm -rf $INSTALL_H_DIR
mkdir -p $INSTALL_H_DIR

if [ ! -d raylib_repo ]; then
  git clone -b master --no-tags https://github.com/commaai/raylib.git raylib_repo
fi

cd raylib_repo

COMMIT=${1:-39e6d8b52db159ba2ab3214b46d89a8069e09394}
git fetch origin $COMMIT
git reset --hard $COMMIT
git clean -xdff .

cd src

make -j$(nproc) PLATFORM=$RAYLIB_PLATFORM RAYLIB_RELEASE_PATH=$INSTALL_DIR
cp raylib.h raymath.h rlgl.h $INSTALL_H_DIR/
echo "raylib development files installed/updated in $INSTALL_H_DIR"

# this commit needs to be in line with raylib
set -x
RAYGUI_COMMIT="76b36b597edb70ffaf96f046076adc20d67e7827"
curl -fsSLo $INSTALL_H_DIR/raygui.h https://raw.githubusercontent.com/raysan5/raygui/$RAYGUI_COMMIT/src/raygui.h

if [ -f /TICI ]; then

  # Building the python bindings
  cd $DIR

  if [ ! -d raylib_python_repo ]; then
    git clone -b master --no-tags https://github.com/commaai/raylib-python-cffi.git raylib_python_repo
  fi

  cd raylib_python_repo

  BINDINGS_COMMIT="ef8141c7979d5fa630ef4108605fc221f07d8cb7"
  git fetch origin $BINDINGS_COMMIT
  git reset --hard $BINDINGS_COMMIT
  git clean -xdff .

  RAYLIB_PLATFORM=$RAYLIB_PLATFORM RAYLIB_INCLUDE_PATH=$INSTALL_H_DIR RAYLIB_LIB_PATH=$INSTALL_DIR python setup.py bdist_wheel
  cd $DIR

  rm -rf wheel
  mkdir wheel
  cp raylib_python_repo/dist/*.whl wheel/

fi
