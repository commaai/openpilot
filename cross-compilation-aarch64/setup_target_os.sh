#!/bin/bash -e

# Dependencies needed to build python 3.8
sudo apt-get update && sudo apt-get install -y --no-install-recommends\
    autoconf \
    build-essential \
    bzip2 \
    capnproto \
    libcapnp-dev \
    clang \
    cmake \
    cppcheck \
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
    libomp-dev \
    libopencv-dev \
    libssl-dev \
    libsqlite3-dev \
    libusb-1.0-0-dev \
    libczmq-dev \
    libzmq3-dev \
    locales \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    opencl-headers \
    python-dev \
    python-pip \
    sudo \
    wget \
    checkinstall \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    zlib1g-dev \
  && sudo rm -rf /var/lib/apt/lists/*

echo -e "\e[1;34m Installing Pyenv... \e[0m"
if ! command -v "pyenv" > /dev/null 2>&1; then
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
fi

echo "export PATH="$HOME/.pyenv/bin:$PATH"" >> ~/.bashrc

PATH="$HOME/.pyenv/bin:$PATH" 
#source ~/.bashrc

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo -e "\e[1;32m Pyenv Installed SUCCESSFULLY!... \e[0m"

# install bashrc
echo -e "\e[1;34m Updating bashrc... \e[0m"
if [ -z "$OPENPILOT_ENV" ]; then
  echo "source $HOME/openpilot/tools/openpilot_env.sh" >> ~/.bashrc
  source ~/.bashrc
  echo -e "\e[1;32m openpilot_env added to bashrc... \e[0m"
fi

# for loggerd to work on ubuntu
# TODO: PC should log somewhere else
sudo mkdir -p /data/media/0/realdata
sudo chown $USER /data/media/0/realdata

# in the openpilot repo
cd $HOME/openpilot

# install python 3.8.2 globally (you should move to python3 anyway)
pyenv install -s 3.8.2 
pyenv global 3.8.2 
pyenv rehash

pip install --upgrade pip

# install pipenv
pip install  --no-cache-dir pipenv
pipenv install --system --deploy --clear 

echo -e "\e[1;32m pipenv setup READY!!...\n\n \e[0m"

cd cereal/
sudo ./install_capnp.sh $ARCH
echo -e "\e[1;32m capnp - INSTALLED!...\n\n \e[0m"

sudo ln -sf $HOME/openpilot/opendbc/can/libdbc.so /lib/aarch64-linux-gnu/libdbc.so
echo -e "\e[1;32m libdbc.so - INSTALLED!...\n\n \e[0m"

echo -e "\e[1;33m ======================================= \e[0m" 
echo -e "\e[1;32m      SETUP COMPLETED SUCCESSFULLY!!!    \e[0m" 
echo -e "\e[1;33m ======================================= \e[0m" 