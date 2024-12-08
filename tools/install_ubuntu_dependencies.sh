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

# Check if stdin is open
if [ -t 0 ]; then
  INTERACTIVE=1
fi

# Install common packages
function install_ubuntu_common_requirements() {
  $SUDO apt-get update
  $SUDO apt-get install -y --no-install-recommends \
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
    libqt5opengl5-dev
}

# Install Ubuntu 24.04 LTS packages
function install_ubuntu_lts_latest_requirements() {
  install_ubuntu_common_requirements

  $SUDO apt-get install -y --no-install-recommends \
    g++-12 \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    python3-dev \
    python3-venv
}

# Install Ubuntu 20.04 packages
function install_ubuntu_focal_requirements() {
  install_ubuntu_common_requirements

  $SUDO apt-get install -y --no-install-recommends \
    libavresample-dev \
    qt5-default \
    python-dev
}

# Detect OS using /etc/os-release file
if [ -f "/etc/os-release" ]; then
  source /etc/os-release
  case "$VERSION_CODENAME" in
    "jammy" | "kinetic" | "noble")
      install_ubuntu_lts_latest_requirements
      ;;
    "focal")
      install_ubuntu_focal_requirements
      ;;
    *)
      echo "$ID $VERSION_ID is unsupported. This setup script is written for Ubuntu 24.04."
      read -p "Would you like to attempt installation anyway? " -n 1 -r
      echo ""
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
      if [ "$UBUNTU_CODENAME" = "focal" ]; then
        install_ubuntu_focal_requirements
      else
        install_ubuntu_lts_latest_requirements
      fi
  esac

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

else
  echo "No /etc/os-release in the system. Make sure you're running on Ubuntu, or similar."
  exit 1
fi
