#!/bin/bash -e

OP_ROOT=$(git rev-parse --show-toplevel)

# Install packages present in all supported versions of Ubuntu
function install_ubuntu_common_requirements() {
  sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    autoconf build-essential clang cmake make cppcheck libtool \
    libstdc++-arm-none-eabi-newlib gcc-arm-none-eabi \
    bzip2 liblzma-dev libarchive-dev libbz2-dev \
    capnproto libcapnp-dev \
    curl libcurl4-openssl-dev wget git git-lfs \
    ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev \
    libeigen3-dev \
    libffi-dev \
    libglew-dev libgles2-mesa-dev libglfw3-dev \
    libglib2.0-0 \
    libomp-dev \
    libopencv-dev \
    libpng16-16 \
    libssl-dev \
    libsqlite3-dev \
    libusb-1.0-0-dev \
    libzmq3-dev \
    libsdl1.2-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
    libportmidi-dev \
    libfreetype6-dev \
    libsystemd-dev \
    locales \
    opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev \
    python-dev python3-pip \
    qml-module-qtquick2 qtmultimedia5-dev qtwebengine5-dev qtlocation5-dev qtpositioning5-dev \
    libqt5sql5-sqlite libqt5svg5-dev libqt5x11extras5-dev \
    libreadline-dev
}

# Install Ubuntu 21.10 packages
function install_ubuntu_latest_requirements() {
  install_ubuntu_common_requirements
  
  sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
}

# Install Ubuntu 20.04 packages
function install_ubuntu_lts_requirements() {
  install_ubuntu_common_requirements

  sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libavresample-dev \
    qt5-default
}

# Detect OS using /etc/os-release file
if [ -f "/etc/os-release" ]; then
  # Pull all variables from /etc/os-release
  eval `cat /etc/os-release`
  if [ "$ID" == "ubuntu" ]; then
    case "$VERSION_ID" in
      "21.10")
        install_ubuntu_latest_requirements
        ;;
      "20.04")
        install_ubuntu_lts_requirements
        ;;
      *)
        echo "Ubuntu version "$VERSION_ID" not supported"
        exit 1
    esac
  else
    echo "OS is not ubuntu"
    exit 1
  fi
else
  echo "No /etc/os-release in the system"
  exit 1
fi


# install pyenv
if ! command -v "pyenv" > /dev/null 2>&1; then
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
fi

# in the openpilot repo
cd $OP_ROOT

source ~/.bashrc
if [ -z "$OPENPILOT_ENV" ]; then
  printf "\nsource %s/tools/openpilot_env.sh" "$OP_ROOT" >> ~/.bashrc
  source ~/.bashrc
  echo "added openpilot_env to bashrc"
fi

# do the rest of the git checkout
git lfs pull
git submodule init
git submodule update

# install python
PYENV_PYTHON_VERSION=$(cat $OP_ROOT/.python-version)
PATH=$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH
pyenv install -s ${PYENV_PYTHON_VERSION}
pyenv global ${PYENV_PYTHON_VERSION}
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
