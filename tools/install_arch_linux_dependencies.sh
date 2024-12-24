#!/usr/bin/env bash
set -e

SUDO=""

# Use sudo if not root
if [[ ! $(id -u) -eq 0 ]]; then
  if [[ -z $(which sudo) ]]; then
    echo "Please install sudo or run as root"
    exit 1
  fi
  SUDO="sudo"
fi

# Install common packages for Arch Linux
function install_arch_common_requirements() {
  $SUDO pacman -Sy --noconfirm \
    ca-certificates \
    clang \
    base-devel \
    arm-none-eabi-gcc \
    lzma \
    capnproto \
    curl \
    git \
    git-lfs \
    ffmpeg \
    libavformat \
    libavcodec \
    libavdevice \
    libavutil \
    libavfilter \
    bzip2 \
    eigen \
    libffi \
    glew \
    mesa \
    glfw \
    glibc \
    qt5-base \
    ncurses \
    openssl \
    libusb \
    zeromq \
    zstd \
    sqlite \
    systemd \
    opencl-headers \
    ocl-icd \
    portaudio \
    qt5-tools \
    qt5-svg \
    qt5-serialbus \
    qt5-x11extras \
    qt5-opengl
}

# Detect OS using /etc/os-release file (optional)
if [ -f "/etc/os-release" ]; then
  source /etc/os-release
  if [[ "$ID" == "arch" ]]; then
      install_arch_common_requirements
  else
      echo "$ID is unsupported. This setup script is written for Arch Linux."
      exit 1
  fi
else
  echo "No /etc/os-release in the system. Make sure you're running on Arch Linux."
  exit 1
fi

# Setup udev rules (if applicable)
if [[ -d "/etc/udev/rules.d/" ]]; then
  # Setup panda udev rules
  $SUDO tee /etc/udev/rules.d/12-panda_jungle.rules > /dev/null <<EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddef", MODE="0666"
EOF

  # Setup jungle udev rules
  $SUDO tee /etc/udev/rules.d/11-panda.rules > /dev/null <<EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddee", MODE="0666"
EOF

  $SUDO udevadm control --reload-rules && $SUDO udevadm trigger || true
fi

echo "All dependencies installed successfully."
