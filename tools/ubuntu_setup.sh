#!/bin/bash -e

# Supported architectures.
AMD64="x86_64"
ARM64="arm64"
SETUP_ERROR=0

if [ "$1" != "" -a "$1" != "-h" -a "$1" != "--help" -a "$1" != "-arch" ]; then
  echo "ubuntu_setup.sh: invalid option $1"
  echo "Try ./ubuntu_setup.sh --help"
  exit 1
fi

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  echo "Usage: ./ubuntu_setup.sh [-arch [x86_64|arm64]]"
  echo ""
  echo "Optional arguments:"
  echo "  -arch    specifies the architecture. Possible values are: x86_64 (default),"
  echo "           arm64."
  exit 0
fi

if [ "$1" == "-arch" ]; then
  if [ "$2" != $AMD64 -a "$2" != $ARM64 ]; then
    echo "ubuntu_setup.sh: invalid value $2 for argument -arch"
    echo "Try ./ubuntu_setup.sh --help"
    exit 1
  else
    ARCH="$2"
  fi
else
  ARCH="x86_64"
fi

#Colors  reset = 0, 
#        black = 30, 
#        red = 31, 
#        green = 32, 
#        yellow = 33, 
#        blue = 34, 
#        magenta = 35, 
#        cyan = 36,
#        white = 37.
echo -e "\e[1;33m Installing Ubuntu dependencies and tools for architecture $ARCH... \e[0m"

UBUNTU_SPECIFIC_DEPS_AMD_64="gcc-arm-none-eabi"
UBUNTU_SPECIFIC_DEPS_ARM_64="libomp5"
if [ $ARCH == $AMD64 ]; then
  UBUNTU_SPECIFIC_DEPS=$UBUNTU_SPECIFIC_DEPS_AMD_64
else
  UBUNTU_SPECIFIC_DEPS=$UBUNTU_SPECIFIC_DEPS_ARM_64
fi

sudo apt-get update && sudo apt-get install -y \
    autoconf \
    build-essential \
    bzip2 \
    capnproto \
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
    libmysqlclient-dev \
    libomp-dev \
    libopencv-dev \
    libpng16-16 \
    libssl-dev \
    libstdc++-arm-none-eabi-newlib \
    libsqlite3-dev \
    libtool \
    libusb-1.0-0-dev \
    libzmq3-dev \
    libczmq-dev \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
    libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev \
    libsystemd-dev \
    locales \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    opencl-headers \
    python-dev \
    python-pip \
    qt5-default \
    screen \
    sudo \
    vim \
    wget \
    scons \
    $UBUNTU_SPECIFIC_DEPS

echo -e "\e[1;32m Ubuntu dependencies installed SUCCESSFULLY!!... \e[0m"

# install git lfs
if ! command -v "git-lfs" > /dev/null 2>&1; then
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
fi

echo -e "\e[1;34m Installing Pyenv... \e[0m"
# install pyenv
if ! command -v "pyenv" > /dev/null 2>&1; then
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
fi

if [ -z "$USER" ]; then
  echo "export PATH="/root/.pyenv/bin:$PATH"" >> /root/.bashrc
  PATH="/root/.pyenv/bin:$PATH"
else
  echo "export PATH="$HOME/.pyenv/bin:$PATH"" >> ~/.bashrc
  PATH="$HOME/.pyenv/bin:$PATH"
fi

#source ~/.bashrc

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo -e "\e[1;32m Pyenv Installed SUCCESSFULLY!... \e[0m"

openpilot_dir="$(dirname "$PWD")"

if [ -z "$OPENPILOT_ENV" ]; then
  if [ -z "$USER" ]; then
    #hack when building inside a docker container and openpilot was not cloned in home directory.
    echo "source $openpilot_dir/tools/openpilot_env.sh" >> /root/.bashrc
  else
    echo "source $openpilot_dir/tools/openpilot_env.sh" >> ~/.bashrc
  fi
  #source ~/.bashrc
  echo -e "\e[1;32m Added openpilot_env to bashrc... \e[0m"
fi

# Workaround for server certificate expiration.
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3EB9326A7BF6DFCD

#install latest CLANG to avoid compilation errors
sudo apt-add-repository "deb [arch=amd64] http://apt.llvm.org/xenial/ llvm-toolchain-xenial main"
sudo apt-get update
sudo apt install -y clang --allow-unauthenticated
echo -e "\e[1;32m Latest CLANG Installed!!... \e[0m"

#Install g++ and gcc 6 - needed for compiling capnp
sudo apt-get update && \
sudo apt-get install build-essential software-properties-common -y && \
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
sudo apt-get update && \
sudo apt-get install gcc-snapshot -y && \
sudo apt-get update && \
sudo apt-get install gcc-6 g++-6 -y && \
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6 && \
sudo apt-get install gcc-4.8 g++-4.8 -y && \
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.8;
echo -e "\e[1;32m g++ and gcc  6 - Installed!!... \e[0m"


# for loggerd to work on ubuntu
# TODO: PC should log somewhere else
#sudo mkdir -p /data/media/0/realdata
#sudo chown $USER /data/media/0/realdata
if [ $ARCH == $ARM64 ]; then

  #multilib install
  sudo apt-get install -y gcc-6-multilib g++-6-multilib --allow-unauthenticated
  sudo apt install -y g++-aarch64-linux-gnu --allow-unauthenticated
  echo -e "\e[1;32m Multilib and g++-aarch64 - Installed!!... \e[0m"

  #linker reference to use a prefix during compilation
  mkdir -p $HOME/linker_bin
  sudo ln -sf /usr/bin/aarch64-linux-gnu-ld $HOME/linker_bin/ld
  echo -e "\e[1;32m Created symbolic link to aarch64 linker... \e[0m"

  # external libraries dependencies
  sudo apt-get update && sudo apt-get install -y -f --allow-unauthenticated \
    libudev-dev:arm64 \
    libjpeg-dev:arm64 \
    libglu1-mesa-dev:arm64 \
    freeglut3-dev:arm64 \
    mesa-common-dev:arm64 \
    ocl-icd-libopencl1:arm64 \
    ocl-icd-opencl-dev:arm64 \
    libzmq3-dev:arm64 \
    libczmq-dbg:arm64 \
    libczmq-dev:arm64 \
    libczmq3:arm64 \
    libavformat-dev:arm64 \
    libswscale-dev:arm64 \
    libz-dev:arm64 \
    libbz2-dev:arm64 \
    libxi-dev:arm64 \
    libglfw3-dev:arm64 \
    ffmpeg:arm64 \
    libsystemd-dev:arm64 \
    qt5-default:arm64 \

  #fix missing installs
  sudo apt-get -f install
  echo -e "\e[1;32m ARM64 dependencies INSTALLED!... \e[0m"

  #create symbolic link for libglfw
  LIBGLFW=/usr/lib/aarch64-linux-gnu/libglfw.so
  sudo ln -sf /usr/lib/aarch64-linux-gnu/libglfw.so.3.1 $LIBGLFW

  if [ -f "$LIBGLFW" ]; then
    echo -e "\e[1;32m --> Symbolic link of libglfw created successfully!! \e[0m"

    cd $openpilot_dir

    #LibUSB cross-compilation and installation for arm64
    sudo ./install_libusb_arm.sh
    echo -e "\e[1;32m LibUSB for ARM - INSTALLED!... \e[0m"

    #FFMPEG cross-compilation and installation for arm64
    sudo ./install_ffmpeg_arm.sh
    echo -e "\e[1;32m FFMpeg for ARM - INSTALLED!... \e[0m"
  else
    echo -e "\e[1;31m ERROR creating libglfw link - cross compilation will fail \e[0m"
    SETUP_ERROR=1
  fi
fi

if [ $SETUP_ERROR == 0 ]; then
  # in the openpilot repo
  cd $openpilot_dir

  # do the rest of the git checkout
  if [-d .git]; then
    git lfs pull
    git submodule init
    git submodule update
  fi

  # install python 3.8.2 globally (you should move to python3 anyway)
  pyenv install -s 3.8.2
  pyenv global 3.8.2
  pyenv rehash

  # **** in python env ****

  # install pipenv
  pip install pipenv==2018.11.26

  # pipenv setup (in openpilot dir)
  pipenv install --dev --system --deploy

  cd $openpilot_dir/cereal/
  sudo ./install_capnp.sh $ARCH

  sudo apt install -y libglfw3-dev

  #create symbolic link for libglfw
  ldconfig

  LIBGLFW=/usr/lib/aarch64-linux-gnu/libglfw.so
  LIBGLFW31=/usr/lib/aarch64-linux-gnu/libglfw.so.3.1
  if [ -f "$LIBGLFW31" ]; then
    sudo ln -sf $LIBGLFW31 $LIBGLFW
    if [ -f "$LIBGLFW" ]; then
        echo "libglfw ok..."
    else
        echo "error with ARM64 libglfw"
    fi
  fi

  unset OPENPILOT_ENV

  echo -e "\e[1;33m ======================================= \e[0m" 
  echo -e "\e[1;32m      SETUP COMPLETED SUCCESSFULLY!!!    \e[0m" 
  echo -e "\e[1;33m ======================================= \e[0m" 
else
  echo -e "\e[1;31m     ERROR while setting up environment  \e[0m" 
fi