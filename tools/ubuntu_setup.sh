#!/bin/bash -e
sudo apt-get update && sudo apt-get install -y \
    autoconf \
    build-essential \
    bzip2 \
    clang \
    cmake \
    curl \
    ffmpeg \
    git \
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
    wget

curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# TODO: add pyenv to .bashrc
pyenv install 3.7.3
pyenv global 3.7.3
pyenv rehash

# install pipenv
pip install pipenv==2018.11.26

# pipenv setup
cd ../
pipenv install --system --deploy

# TODO: add openpilot to PYTHONPATH and external to PATH, this should be in bashrc
# export PYTHONPATH="$HOME/openpilot"
# export PATH="$PATH:$HOME/openpilot/external/capnp/bin"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/openpilot/external/capnp/lib"

# TODO: run external/capnp/build.sh ... needed?

# at this point, manager runs

# to make tools work
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev

pip install -r tools/requirements.txt

