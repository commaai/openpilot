#!/bin/bash -e

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    autoconf \
    build-essential \
    bzip2 \
    capnproto \
    cppcheck \
    libcapnp-dev \
    clang \
    cmake \
    curl \
    ffmpeg \
    git \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev \
    libarchive-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libffi-dev \
    libglew-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglib2.0-0 \
    liblzma-dev \
    libomp-dev \
    libopencv-dev \
    libpng16-16 \
    libssl-dev \
    libstdc++-arm-none-eabi-newlib \
    libsqlite3-dev \
    libtool \
    libusb-1.0-0-dev \
    libzmq3-dev \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
    libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev \
    libsystemd-dev \
    locales \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    opencl-headers \
    python-dev \
    python3-pip \
    qml-module-qtquick2 \
    qt5-default \
    qtmultimedia5-dev \
    qtwebengine5-dev \
    qtlocation5-dev \
    qtpositioning5-dev \
    libqt5sql5-sqlite \
    libqt5svg5-dev \
    screen \
    sudo \
    vim \
    wget \
    gcc-arm-none-eabi \
    libqt5x11extras5-dev \
    libreadline-dev

# install git lfs
if ! command -v "git-lfs" > /dev/null 2>&1; then
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
fi

# install pyenv
if ! command -v "pyenv" > /dev/null 2>&1; then
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
fi

# in the openpilot repo
TOOLS_DIR="$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$(dirname $TOOLS_DIR)"

source ~/.bashrc
if [ -z "$OPENPILOT_ENV" ]; then
  OP_DIR=$(git rev-parse --show-toplevel)
  printf "\nsource %s/tools/openpilot_env.sh" "$OP_DIR" >> ~/.bashrc
  source ~/.bashrc
  echo "added openpilot_env to bashrc"
fi

# do the rest of the git checkout
git lfs pull
git submodule init
git submodule update

# install python
PATH=$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH
pyenv install -s 3.8.5
pyenv global 3.8.5
pyenv rehash
eval "$(pyenv init -)"

# **** in python env ****
pip install --upgrade pip==20.2.4
pip install pipenv==2020.8.13
pipenv install --dev --system --deploy

echo
echo "----   FINISH OPENPILOT SETUP   ----"
echo "Configure your active shell env by running:"
echo "source ~/.bashrc"
