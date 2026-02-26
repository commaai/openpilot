#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(cd "$DIR/../" && pwd)"

function detect_sudo() {
  SUDO=""

  if [[ ! $(id -u) -eq 0 ]]; then
    if ! command -v sudo > /dev/null 2>&1; then
      echo "Please install sudo or run as root"
      exit 1
    fi
    SUDO="sudo"
  fi
}

function install_system_deps() {
  detect_sudo

  if [ ! -f "/etc/os-release" ]; then
    echo "No /etc/os-release found. Cannot detect distribution."
    exit 1
  fi
  source /etc/os-release

  # Detect distro family from $ID and $ID_LIKE
  local family=""
  for id in $ID $ID_LIKE; do
    case "$id" in
      ubuntu|debian|linuxmint|pop|zorin|elementary|neon) family="apt";    break ;;
      fedora|rhel|centos|almalinux|rocky)                family="dnf";    break ;;
      arch|manjaro|endeavouros|cachyos|garuda)           family="pacman"; break ;;
      opensuse*|sles)                                    family="zypper"; break ;;
    esac
  done

  if [[ -z "$family" ]]; then
    echo "Unsupported distribution: ${PRETTY_NAME:-$ID}. Supported families: apt (Debian/Ubuntu), dnf (Fedora/RHEL), pacman (Arch), zypper (openSUSE)."
    exit 1
  fi

  case "$family" in
    apt)    $SUDO apt-get update && $SUDO apt-get install -y --no-install-recommends ca-certificates build-essential curl file libcurl4-openssl-dev zlib1g-dev locales git xvfb libgl1-mesa-dri ;;
    dnf)    $SUDO dnf install -y gcc gcc-c++ make ca-certificates curl file git libcurl-devel zlib-devel libubsan glibc-langpack-en xorg-x11-server-Xvfb mesa-dri-drivers ;;
    pacman) $SUDO pacman -Syu --noconfirm base-devel ca-certificates curl file git zlib xorg-server-xvfb mesa ;;
    zypper) $SUDO zypper install -y gcc gcc-c++ make ca-certificates curl file git libcurl-devel zlib-devel glibc-locale xorg-x11-server-Xvfb Mesa-dri ;;
  esac
}

function install_udev_rules() {
  if [[ -d "/etc/udev/rules.d/" ]]; then
    detect_sudo

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
  install_system_deps
  install_udev_rules
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
