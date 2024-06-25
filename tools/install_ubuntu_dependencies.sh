#!/usr/bin/env bash
set -e

_RUNTIME_PACKAGES="
    ca-certificates
    clang
    cppcheck
    build-essential
    gcc-arm-none-eabi
    liblzma-dev
    capnproto
    libcapnp-dev
    curl
    libcurl4-openssl-dev
    git
    git-lfs
    ffmpeg
    libavformat-dev
    libavcodec-dev
    libavdevice-dev
    libavutil-dev
    libavfilter-dev
    libbz2-dev
    libeigen3-dev
    libffi-dev
    libglew-dev
    libgles2-mesa-dev
    libglfw3-dev
    libglib2.0-0
    libqt5charts5-dev
    libncurses5-dev
    libssl-dev
    libusb-1.0-0-dev
    libzmq3-dev
    libsqlite3-dev
    libsystemd-dev
    locales
    opencl-headers
    ocl-icd-libopencl1
    ocl-icd-opencl-dev
    portaudio19-dev
    qml-module-qtquick2
    qtmultimedia5-dev
    qtdeclarative5-dev
    qttools5-dev-tools
    libqt5svg5-dev
    libqt5serialbus5-dev
    libqt5x11extras5-dev
    libqt5opengl5-dev
    "

_DEV_PACKAGES="
    cmake
    make
    clinfo
    "

_EXTRA_PACKAGES="
    casync
    libqt5sql5-sqlite
    libreadline-dev
    libdw1
    autoconf
    libtool
    bzip2
    libarchive-dev
    libncursesw5-dev
    libportaudio2
    locales
    "

_NOBLE_PACKAGES="
    g++-12
    qtbase5-dev
    qtchooser
    qt5-qmake
    qtbase5-dev-tools
    python3-dev
    python3-venv
    "

_FOCAL_PACKAGES="
    libavresample-dev
    qt5-default
    python-dev
    "

SUDO=""

# Use sudo if not root
if [[ ! $(id -u) -eq 0 ]]; then
  if [[ -z $(which sudo) ]]; then
    echo "Please install sudo or run as root"
    exit 1
  fi
  SUDO="sudo"
fi

function apt_install () {
  $SUDO apt-get install -y --no-install-recommends $1
}

RUNTIME_PACKAGES=""
OS_PACKAGES=""
DEV_PACKAGES=""
EXTRA_PACKAGES=""

function installer () {
  if [[ -z "$INTERACTIVE" ]]; then
    echo ""
    printf '=%.0s' $(seq 1 ${#2})
    echo ""

    echo "$1"
    read -p "$2" -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      return 0
    fi
  fi
  apt_install "$1"
}

function install_packages() {
  $SUDO apt-get update

  installer "$RUNTIME_PACKAGES" "Do you want to install these runtime packages? (recommended) [Y/n]: "
  installer "$OS_PACKAGES" "Do you want to install these os-specific packages? (recommended) [Y/n]: "
  installer "$DEV_PACKAGES" "Do you want to install these development packages? [Y/n]: "
  if [[ -z "$INSTALL_EXTRA_PACKAGES" ]]; then
    installer "$EXTRA_PACKAGES" "Do you want to install these extra packages? [Y/n]: "
  fi
}

function get_noble_packages() {
  RUNTIME_PACKAGES=$_RUNTIME_PACKAGES
  OS_PACKAGES=$_NOBLE_PACKAGES
  DEV_PACKAGES=$_DEV_PACKAGES
  EXTRA_PACKAGES=$_EXTRA_PACKAGES
}

function get_focal_packages() {
  RUNTIME_PACKAGES=$_RUNTIME_PACKAGES
  OS_PACKAGES=$_FOCAL_PACKAGES
  DEV_PACKAGES=$_DEV_PACKAGES
  EXTRA_PACKAGES=$_EXTRA_PACKAGES
}

# Detect OS using /etc/os-release file
if [ -f "/etc/os-release" ]; then
  source /etc/os-release
  case "$VERSION_CODENAME" in
    "jammy" | "kinetic" | "noble")
      get_noble_packages
      ;;
    "focal")
      get_focal_packages
      ;;
    *)
      echo "$ID $VERSION_ID is unsupported. This setup script is written for Ubuntu 24.04."
      read -p "Would you like to attempt installation anyway? " -n 1 -r
      echo ""
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
      if [ "$UBUNTU_CODENAME" = "focal" ]; then
        get_focal_packages
      else
        get_noble_packages
      fi
  esac

  install_packages

else
  echo "No /etc/os-release in the system. Make sure you're running on Ubuntu, or similar."
  exit 1
fi
