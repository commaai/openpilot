#!/bin/bash

# Add the apt-fast repository
sudo add-apt-repository ppa:apt-fast/stable -y
sudo apt-get update

# Install apt-fast
sudo apt-get install -y apt-fast

# Configure apt-fast
export _MAXNUM=100
export _APTFAST_CONF=/etc/apt-fast.conf
echo "_MAXNUM=16" | sudo tee -a $_APTFAST_CONF
echo "_DOWNLOADER='aria2c'" | sudo tee -a $_APTFAST_CONF  # Use aria2c for optimized downloads

# Install packages with apt-fast
sudo apt-fast install -y --no-install-recommends \
    ca-certificates \
    clang \
    cppcheck \
    build-essential \
    gcc-arm-none-eabi \
    liblzma-dev \
    capnproto \
    libcapnp-dev \
    curl \
    libcurl4-openssl-dev \
    git \
    git-lfs \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libbz2-dev \
    libeigen3-dev \
    libffi-dev \
    libglew-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglib2.0-0 \
    libqt5charts5-dev \
    libncurses5-dev \
    libssl-dev \
    libusb-1.0-0-dev \
    libzmq3-dev \
    libzstd-dev \
    libsqlite3-dev \
    libsystemd-dev \
    locales \
    opencl-headers \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    portaudio19-dev \
    qttools5-dev-tools \
    libqt5svg5-dev \
    libqt5serialbus5-dev  \
    libqt5x11extras5-dev \
    libqt5opengl5-dev \
    x11-xserver-utils \
    xvfb \
    pulseaudio \
    g++-12 \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools

# Install uv tools
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --all-extras
