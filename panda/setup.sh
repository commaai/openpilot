#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

PLATFORM=$(uname -s)

echo "installing dependencies"
if [[ $PLATFORM == "Darwin" ]]; then
  export HOMEBREW_NO_AUTO_UPDATE=1
  brew install --cask gcc-arm-embedded
  brew install python3 gcc@13
elif [[ $PLATFORM == "Linux" ]]; then
  sudo apt-get install -y --no-install-recommends \
    curl \
    make g++ git libnewlib-arm-none-eabi \
    libusb-1.0-0 \
    gcc-arm-none-eabi python3-pip python3-venv python3-dev
else
  echo "WARNING: unsupported platform. skipping apt/brew install."
fi

if ! command -v uv &>/dev/null; then
  echo "'uv' is not installed. Installing 'uv'..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # doesn't require sourcing on all platforms
  set +e
  source $HOME/.local/bin/env
  set -e
fi

export UV_PROJECT_ENVIRONMENT="$DIR/.venv"
uv sync --all-extras
source "$DIR/.venv/bin/activate"
