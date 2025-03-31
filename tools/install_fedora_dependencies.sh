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
function install_fedora_common_requirements() {
  $SUDO dnf -y update
  $SUDO dnf -y group install "development-tools"
  $SUDO dnf install -y \
	ca-certificates \
	clang \
	capnproto \
	curl \
	git \
	vim \
	git-lfs \
	ffmpeg \
	opencl-headers \
	openssl \
	openssl-devel \
        libcurl-devel \
	libusb \
	ocl-icd-devel \
	qt-devel \
	qtchooser \
        python-devel \
	portaudio-devel \
	gcc-arm-linux-gnu
}

# Install Fedora latest packages
function install_fedora_latest_requirements() {
  install_fedora_common_requirements

  $SUDO dnf update -y 
}

# Detect OS using /etc/os-release file
if [ -f "/etc/os-release" ]; then
  source /etc/os-release
  case "$ID" in
    "fedora" | "kinetic" | "noble")
      install_fedora_latest_requirements
      ;;
    *)
      echo "$ID $VERSION_ID is unsupported. This setup script is written for Ubuntu 24.04."
      read -p "Would you like to attempt installation anyway? " -n 1 -r
      echo ""
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
      install_fedora_latest_requirements
  esac

  if [[ -d "/etc/udev/rules.d/" ]]; then
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
  fi

else
  echo "No /etc/os-release in the system. Make sure you're running on Fedora, or similar."
  exit 1
fi
