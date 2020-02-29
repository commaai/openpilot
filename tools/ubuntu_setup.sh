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
    wget

# git lfs to pull models
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# in the openpilot repo
cd $HOME/openpilot
git lfs pull
git submodule init
git submodule update

# install pyenv
if [ ! -d $HOME/.pyenv ]; then
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
fi

# install bashrc
source ~/.bashrc
if [ -z "$OPENPILOT_ENV" ]; then
  echo "source $HOME/openpilot/tools/openpilot_env.sh" >> ~/.bashrc
  source ~/.bashrc
  echo "added openpilot_env to bashrc"
fi

# install python 3.7.3 globally
pyenv install -s 3.7.3
pyenv global 3.7.3
pyenv rehash

# install pipenv
pip install pipenv==2018.11.26

# pipenv setup (in openpilot dir)
pipenv install --system --deploy

# install capnp (not needed anymore)
#cd external/capnp
#if [ ! -d lib ]; then
#  ./build.sh
#  git checkout bin/*   # don't update these
#fi
#cd ../../

# at this point, manager runs

# to make tools work
pip install -r tools/requirements.txt

# to make modeld work on PC with nvidia GPU
pip install tensorflow-gpu==2.0

# for loggerd to work on ubuntu
# TODO: PC should log somewhere else
#sudo mkdir -p /data/media/0/realdata
#sudo chown $USER /data/media/0/realdata

