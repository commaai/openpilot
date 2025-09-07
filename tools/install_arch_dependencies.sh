#!/usr/bin/env bash
set -euo pipefail

SUDO=""
PACMAN_OPTS="--needed"
INTERACTIVE=0

# Use sudo if not root
if [[ "$(id -u)" -ne 0 ]]; then
	if ! command -v sudo >/dev/null 2>&1; then
		echo "Please install sudo or run as root"
		exit 1
	fi
	SUDO="sudo"
fi

# Check if stdin is open (interactive)
if [ -t 0 ]; then
	INTERACTIVE=1
fi

# Non-interactive pacman opts
if [[ "$INTERACTIVE" -eq 0 ]]; then
	PACMAN_OPTS="$PACMAN_OPTS --noconfirm --needed"
fi

install_arch_requirements() {

  git lfs install || true

	# Core toolchains & libs
	$SUDO pacman -S $PACMAN_OPTS \
		ca-certificates \
		clang \
		base-devel \
		arm-none-eabi-gcc arm-none-eabi-binutils arm-none-eabi-newlib \
		xz \
		capnproto \
		curl \
		git git-lfs \
		ffmpeg \
		bzip2 \
		eigen \
		libffi \
		glew \
		mesa \
		glfw \
		glib2 \
		libjpeg-turbo \
		qt5-base qt5-tools qt5-svg qt5-x11extras qt5-charts \
		ncurses \
		openssl \
		libusb \
		zeromq \
		zstd \
		sqlite \
		systemd-libs \
		opencl-headers ocl-icd \
		portaudio \
		xorg-server-xvfb \
		python python-pip \
    arm-none-eabi-newlib

    yay -S qt5-serialbus
}

# Detect OS using /etc/os-release
if [[ -f "/etc/os-release" ]]; then
	. /etc/os-release
	if [[ "${ID:-}" == "arch"  ]]; then
		install_arch_requirements
	else
		echo "$ID $VERSION_ID is unsupported here. This setup script is written for Arch-based systems."
		if [[ "$INTERACTIVE" -eq 1 ]]; then
			read -p "Attempt installation anyway? [y/N] " -r
			echo ""
			if [[ ! $REPLY =~ ^[Yy]$ ]]; then
				exit 1
			fi
			install_arch_requirements
		else
			exit 1
		fi
	fi

	# udev rules
	if [[ -d "/etc/udev/rules.d/" ]]; then
		$SUDO tee /etc/udev/rules.d/12-panda_jungle.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddef", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcf", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddef", MODE="0666"
EOF

		$SUDO tee /etc/udev/rules.d/11-panda.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddee", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcc", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddee", MODE="0666"
EOF

		$SUDO udevadm control --reload-rules && $SUDO udevadm trigger || true
	fi
else
	echo "No /etc/os-release found. Make sure you're on an Arch-based system."
	exit 1
fi
