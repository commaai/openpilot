#!/bin/bash -e

# NOTE: ubuntu_setup.sh doesn't run! only for reading now

sudo apt-get update && sudo apt-get install -y \
    autoconf \
    build-essential \
    bzip2 \
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
    libmysqlclient-dev \
    libomp-dev \
    libopencv-dev \
    libpng16-16 \
    libssl-dev \
    libstdc++-arm-none-eabi-newlib \
    libsqlite3-dev \
    libtool \
    libusb-1.0-0-dev \
    libzmq5-dev \
    locales \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    opencl-headers \
    python-dev \
    python-pip \
    screen \
    sudo \
    vim \
    wget \
    gcc-arm-none-eabi

# install git lfs
if ! command -v "git-lfs" > /dev/null 2>&1; then
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
fi

# install pyenv
if ! command -v "pyenv" > /dev/null 2>&1; then
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
fi

# install bashrc
source ~/.bashrc
if [ -z "$OPENPILOT_ENV" ]; then
  echo "source $HOME/openpilot/tools/openpilot_env.sh" >> ~/.bashrc
  source ~/.bashrc
  echo "added openpilot_env to bashrc"
fi

# in the openpilot repo
cd $HOME/openpilot

# do the rest of the git checkout
git lfs pull
git submodule init
git submodule update

# install python 3.7.3 globally (you should move to python3 anyway)
pyenv install -s 3.7.3
pyenv global 3.7.3
pyenv rehash

# **** in python env ****

# install pipenv
pip install pipenv==2018.11.26

# pipenv setup (in openpilot dir)
pipenv install --system --deploy

# to make tools work
pip install -r tools/requirements.txt

# to make modeld work on PC with nvidia GPU
pip install tensorflow-gpu==2.1

# for loggerd to work on ubuntu
# TODO: PC should log somewhere else
#sudo mkdir -p /data/media/0/realdata
#sudo chown $USER /data/media/0/realdata
