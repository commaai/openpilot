#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(cd "$DIR/../" && pwd)"

function install_ubuntu_deps() {
  SUDO=""

  if [[ ! $(id -u) -eq 0 ]]; then
    if [[ -z $(which sudo) ]]; then
      echo "Please install sudo or run as root"
      exit 1
    fi
    SUDO="sudo"
  fi

  # Detect OS using /etc/os-release file
  if [ -f "/etc/os-release" ]; then
    source /etc/os-release
    case "$VERSION_CODENAME" in
      "jammy" | "kinetic" | "noble")
        ;;
      *)
        echo "$ID $VERSION_ID is unsupported. This setup script is written for Ubuntu 24.04."
        read -p "Would you like to attempt installation anyway? " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
          exit 1
        fi
        ;;
    esac
  else
    echo "No /etc/os-release in the system. Make sure you're running on Ubuntu, or similar."
    exit 1
  fi

  $SUDO apt-get update

  # normal stuff, mostly for the bare docker image
  $SUDO apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    libssl-dev \
    libcurl4-openssl-dev \
    locales \
    git \
    xvfb

  $SUDO apt-get install -y --no-install-recommends \
    python3-dev \
    libncurses5-dev \
    libzstd-dev

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

    # Setup adb udev rules
    $SUDO tee /etc/udev/rules.d/50-comma-adb.rules > /dev/null <<EOF
SUBSYSTEM=="usb", ATTR{idVendor}=="04d8", ATTR{idProduct}=="1234", ENV{adb_user}="yes"
EOF

    $SUDO udevadm control --reload-rules && $SUDO udevadm trigger || true
  fi
}

function install_python_deps() {
  # Increase the pip timeout to handle TimeoutError
  export PIP_DEFAULT_TIMEOUT=200

  cd "$ROOT"

  if ! command -v "uv" > /dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf --retry 5 --retry-delay 5 --retry-all-errors https://astral.sh/uv/install.sh | sh
    UV_BIN="$HOME/.local/bin"
    PATH="$UV_BIN:$PATH"
  fi

  echo "updating uv..."
  # ok to fail, can also fail due to installing with brew
  uv self update || true

  echo "installing python packages..."
  uv sync --frozen --all-extras
  source .venv/bin/activate

  if [[ "$(uname)" == 'Darwin' ]]; then
    touch "$ROOT"/.env
    echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> "$ROOT"/.env
  fi
}

# --- Main ---

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  install_ubuntu_deps
  echo "[ ] installed system dependencies t=$SECONDS"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ $SHELL == "/bin/zsh" ]]; then
    RC_FILE="$HOME/.zshrc"
  elif [[ $SHELL == "/bin/bash" ]]; then
    RC_FILE="$HOME/.bash_profile"
  fi
fi

if [ -f "$ROOT/pyproject.toml" ]; then
  install_python_deps
  echo "[ ] installed python dependencies t=$SECONDS"
fi

if [[ "$OSTYPE" == "darwin"* ]] && [[ -n "${RC_FILE:-}" ]]; then
  echo
  echo "----   OPENPILOT SETUP DONE   ----"
  echo "Open a new shell or configure your active shell env by running:"
  echo "source $RC_FILE"
fi
