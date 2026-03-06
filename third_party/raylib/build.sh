#!/usr/bin/env bash
set -e

export SOURCE_DATE_EPOCH=0
export ZERO_AR_DATE=1

SUDO=""

# Use sudo if not root
if [[ ! $(id -u) -eq 0 ]]; then
  if [[ -z $(which sudo) ]]; then
    echo "Please install sudo or run as root"
    exit 1
  fi
  SUDO="sudo"
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

RAYLIB_PLATFORM="PLATFORM_DESKTOP"

ARCHNAME=$(uname -m)
if [ -f /TICI ]; then
  ARCHNAME="larch64"
  RAYLIB_PLATFORM="PLATFORM_COMMA"
elif [[ "$OSTYPE" == "linux"* ]]; then
  # required dependencies on Linux PC
  $SUDO apt install \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev
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

COMMIT=${1:-3425bd9d1fb292ede4d80f97a1f4f258f614cffc}
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

  BINDINGS_COMMIT="a0710d95af3c12fd7f4b639589be9a13dad93cb6"
  git fetch origin $BINDINGS_COMMIT
  git reset --hard $BINDINGS_COMMIT
  git clean -xdff .

  RAYLIB_PLATFORM=$RAYLIB_PLATFORM RAYLIB_INCLUDE_PATH=$INSTALL_H_DIR RAYLIB_LIB_PATH=$INSTALL_DIR python setup.py bdist_wheel
  cd $DIR

  rm -rf wheel
  mkdir wheel
  cp raylib_python_repo/dist/*.whl wheel/

fi
