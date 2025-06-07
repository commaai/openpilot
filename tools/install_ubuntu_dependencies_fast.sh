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

# Optimized package installation for CI
function install_ubuntu_fast_requirements() {
  echo "Installing Ubuntu dependencies (optimized for CI)..."

  # Use dpkg-query to check if critical packages are installed
  if dpkg-query -W clang build-essential python3-dev &>/dev/null && [ "$CI" == "1" ]; then
    echo "Critical packages already installed, skipping full install..."
    return 0
  fi

  # Skip update if packages were recently updated (for CI caching)
  if [ ! -f "/var/lib/apt/lists/lock" ] || [ ! "$(find /var/lib/apt/lists -name '*.ubuntu.com_*' -mtime -1 2>/dev/null)" ]; then
    echo "Updating package lists..."
    $SUDO apt-get update
  else
    echo "Package lists are recent, skipping update..."
  fi

  # Install packages with optimizations for CI - parallel downloads
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    --allow-unauthenticated \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    -o APT::Install-Suggests=false \
    -o APT::Install-Recommends=false \
    ca-certificates \
    clang \
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
    libjpeg-dev \
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
    libqt5serialbus5-dev \
    libqt5x11extras5-dev \
    libqt5opengl5-dev \
    xvfb \
    g++-12 \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    python3-dev \
    python3-venv

  echo "✅ Ubuntu dependencies installed successfully"
}

# Skip udev rules setup in CI (not needed)
function setup_udev_rules() {
  if [[ "$CI" == "1" ]]; then
    echo "Skipping udev rules setup in CI environment"
    return 0
  fi

  if [[ -d "/etc/udev/rules.d/" ]]; then
    echo "Setting up udev rules..."
    # Setup jungle udev rules
    $SUDO tee /etc/udev/rules.d/12-panda_jungle.rules > /dev/null <<EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddef", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddef", MODE="0666"
EOF

    # Setup panda udev rules
    $SUDO tee /etc/udev/rules.d/11-panda.rules > /dev/null <<EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddee", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddee", MODE="0666"
EOF

    $SUDO udevadm control --reload-rules && $SUDO udevadm trigger || true
    echo "✅ Udev rules configured"
  fi
}

# Detect OS using /etc/os-release file
if [ -f "/etc/os-release" ]; then
  source /etc/os-release
  case "$VERSION_CODENAME" in
    "jammy" | "kinetic" | "noble")
      install_ubuntu_fast_requirements
      setup_udev_rules
      ;;
    *)
      echo "$ID $VERSION_ID is unsupported. This setup script is written for Ubuntu 24.04."
      if [[ "$CI" != "1" ]]; then
        read -p "Would you like to attempt installation anyway? " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
          exit 1
        fi
      fi
      install_ubuntu_fast_requirements
      setup_udev_rules
  esac
else
  echo "No /etc/os-release in the system. Make sure you're running on Ubuntu, or similar."
  exit 1
fi

echo "✅ Fast Ubuntu dependencies installation completed!"