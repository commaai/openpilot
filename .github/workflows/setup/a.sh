#!/bin/bash
set -ex

# Environment setup
export PYTHONUNBUFFERED=1
export DEBIAN_FRONTEND=noninteractive

# Base dependencies
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    sudo tzdata locales ssh pulseaudio xvfb \
    x11-xserver-utils gnome-screenshot python3-tk python3-dev

# Locale configuration
sudo sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8

# Install project-specific Ubuntu dependencies
chmod +x tools/install_ubuntu_dependencies.sh
sudo ./tools/install_ubuntu_dependencies.sh
sudo rm -rf /var/lib/apt/lists/* /tmp/*

# Remove unnecessary GCC components
sudo rm -rf /usr/lib/gcc/arm-none-eabi/*/{arm,thumb/nofp,thumb/v6*,thumb/v8*,thumb/v7*}

# OpenCL dependencies
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    alien unzip tar curl xz-utils dbus \
    gcc-arm-none-eabi tmux vim libx11-6 wget

# Intel OpenCL driver setup
mkdir -p /tmp/opencl-driver-intel
pushd /tmp/opencl-driver-intel
wget https://github.com/intel/llvm/releases/download/2024-WW14/oclcpuexp-2024.17.3.0.09_rel.tar.gz
wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.12.0/oneapi-tbb-2021.12.0-lin.tgz

sudo mkdir -p /opt/intel/oclcpuexp_2024.17.3.0.09_rel
sudo tar -zxvf oclcpuexp-2024.17.3.0.09_rel.tar.gz -C /opt/intel/oclcpuexp_2024.17.3.0.09_rel
echo "/opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64/libintelocl.so"

sudo tar -zxvf oneapi-tbb-2021.12.0-lin.tgz -C /opt/intel
sudo ln -s /opt/intel/oneapi-tbb-2021.12.0/lib/intel64/gcc4.8/libtbb* /opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64/

echo "/opt/intel/oclcpuexp_2024.17.3.0.09_rel/x64" | sudo tee /etc/ld.so.conf.d/libintelopenclexp.conf
sudo ldconfig
popd
sudo rm -rf /tmp/opencl-driver-intel

# Environment variables
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
export QTWEBENGINE_DISABLE_SANDBOX=1

# DBUS setup
sudo dbus-uuidgen > /etc/machine-id

# Python environment setup
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# Install Python dependencies
chmod +x tools/install_python_dependencies.sh
./tools/install_python_dependencies.sh

# Git safety override
git config --global --add safe.directory /tmp/openpilot

echo "CI setup completed successfully"
