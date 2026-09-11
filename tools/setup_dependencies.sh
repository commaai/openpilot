#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(git -C "$DIR" rev-parse --show-toplevel)"

VENV_BIN="bin"
case "$(uname -s)" in MINGW*|MSYS*) VENV_BIN="Scripts" ;; esac  # native Python venv layout on Windows

function retry() {
  local attempts=$1
  shift
  for i in $(seq 1 "$attempts"); do
    if "$@"; then
      return 0
    fi
    if [ "$i" -lt "$attempts" ]; then
      echo "  Attempt $i/$attempts failed, retrying in 5s..."
      sleep 5
    fi
  done
  return 1
}

function install_linux_deps() {
  SUDO=""

  if [[ ! $(id -u) -eq 0 ]]; then
    if [[ -z $(which sudo) ]]; then
      echo "Please install sudo or run as root"
      exit 1
    fi
    SUDO="sudo"
  fi

  local missing_linux_deps=0
  for cmd in gcc g++ make curl curl-config git; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
      missing_linux_deps=1
      break
    fi
  done

  # ------------------------------------------------
  # dependencies should never be added to this list.
  # these are only for inflating bare docker images
  # to their desktop equivalents.
  #
  # from a real desktop OS image, we install uv
  # then use that to install all the dependencies
  # specified in pyproject.toml.
  # ------------------------------------------------
  if [[ "$missing_linux_deps" -eq 0 ]]; then
    # the native package managers are slow, so skip if we can
    echo "[ ] system packages already installed t=$SECONDS"
  elif command -v apt-get > /dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends ca-certificates build-essential curl libcurl4-openssl-dev locales git xclip wl-clipboard
  elif command -v dnf > /dev/null 2>&1; then
    $SUDO dnf install -y ca-certificates gcc gcc-c++ make curl libcurl-devel glibc-langpack-en git
  elif command -v yum > /dev/null 2>&1; then
    $SUDO yum install -y ca-certificates gcc gcc-c++ make curl libcurl-devel glibc-langpack-en git
  elif command -v pacman > /dev/null 2>&1; then
    $SUDO pacman -Syu --noconfirm --needed base-devel ca-certificates curl git
  elif command -v zypper > /dev/null 2>&1; then
    $SUDO zypper --non-interactive refresh
    $SUDO zypper --non-interactive install ca-certificates gcc gcc-c++ make curl libcurl-devel glibc-locale git
  elif command -v apk > /dev/null 2>&1; then
    $SUDO apk add --no-cache ca-certificates build-base curl curl-dev musl-locales git
  elif command -v xbps-install > /dev/null 2>&1; then
    $SUDO xbps-install -Syu base-devel ca-certificates curl git libcurl-devel glibc-locales
  else
    echo "Unsupported Linux distribution. Supported package managers: apt-get, dnf, yum, pacman, zypper, apk, xbps-install."
    exit 1
  fi

  if [[ -d "/etc/udev/rules.d/" ]]; then
    $SUDO tee /etc/udev/rules.d/11-openpilot.rules > /dev/null <<-EOF
	# Panda Jungle devices
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcf", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddef", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcf", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddef", MODE="0666"

	# Panda devices
	SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcc", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddee", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcc", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddee", MODE="0666"

	# comma devices over ADB
	SUBSYSTEM=="usb", ATTR{idVendor}=="04d8", ATTR{idProduct}=="1234", ENV{adb_user}="yes"
	EOF

    # delete the old ones
    $SUDO rm -f /etc/udev/rules.d/11-panda.rules /etc/udev/rules.d/12-panda_jungle.rules /etc/udev/rules.d/50-comma-adb.rules

    $SUDO udevadm control --reload-rules && $SUDO udevadm trigger || true
  fi
}

function install_windows_deps() {
  [[ "${MSYSTEM:-}" == "CLANG64" ]] || { echo "Windows builds need an MSYS2 CLANG64 shell, this is ${MSYSTEM:-not MSYS2}"; exit 1; }
  # pycapnp's wheel links the MSVC C++ runtime, which neither Python nor a fresh Windows ships
  [[ -f "$(cygpath -u "$SYSTEMROOT")/System32/msvcp140.dll" ]] || { echo "install the Visual C++ Redistributable https://aka.ms/vc14/vc_redist.x64.exe and rerun"; exit 1; }
  # clang/lld/libc++ (MSVC cannot build openpilot's GNU C), dlfcn, MSYS2's native git and uv (the venv's git-lfs mis-resolves msys git's POSIX paths)
  pacman -S --needed --noconfirm \
    "$MINGW_PACKAGE_PREFIX-toolchain" "$MINGW_PACKAGE_PREFIX-pkgconf" "$MINGW_PACKAGE_PREFIX-ccache" \
    "$MINGW_PACKAGE_PREFIX-dlfcn" "$MINGW_PACKAGE_PREFIX-git" "$MINGW_PACKAGE_PREFIX-uv" file
}

function install_python_deps() {
  # Increase the pip timeout to handle TimeoutError
  export PIP_DEFAULT_TIMEOUT=200

  cd "$ROOT"

  if ! command -v "uv" > /dev/null 2>&1; then
    echo "installing uv..."
    # TODO: outer retry can be removed once https://github.com/axodotdev/cargo-dist/pull/2311 is merged
    retry 3 sh -c 'curl --retry 5 --retry-delay 5 --retry-all-errors -LsSf https://astral.sh/uv/install.sh | UV_GITHUB_TOKEN="${GITHUB_TOKEN:-}" sh'
    UV_BIN="$HOME/.local/bin"
    PATH="$UV_BIN:$PATH"
  fi

  echo "updating uv..."
  # ok to fail, can also fail due to installing with brew
  uv self update || true

  echo "installing python packages..."
  uv sync --frozen --all-extras
  [[ $VENV_BIN == bin ]] || cp -n .venv/Scripts/python.exe .venv/Scripts/python3.exe  # scons commands and shebangs say python3
  source .venv/$VENV_BIN/activate
}

# --- Main ---

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  install_linux_deps
  echo "[ ] installed system dependencies t=$SECONDS"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ $SHELL == "/bin/zsh" ]]; then
    RC_FILE="$HOME/.zshrc"
  elif [[ $SHELL == "/bin/bash" ]]; then
    RC_FILE="$HOME/.bash_profile"
  fi
elif [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  install_windows_deps
  echo "[ ] installed system dependencies t=$SECONDS"
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
