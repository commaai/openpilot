#!/bin/bash
set -e

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# TODO: why doesn't uv do this?
export PYTHONPATH=$BASEDIR

# *** dependencies install ***
if [ "$(uname -s)" = "Linux" ]; then
  if ! command -v "mull-runner-18" > /dev/null 2>&1; then
    curl -1sLf 'https://dl.cloudsmith.io/public/mull-project/mull-stable/setup.deb.sh' | sudo -E bash
    sudo apt-get update && sudo apt-get install -y clang-18 mull-18
  fi
elif [ "$(uname -s)" = "Darwin" ]; then
  if ! brew list llvm@18 &>/dev/null; then
    brew install llvm@18
  fi
  if [ ! -f "$BASEDIR/.mull/bin/mull-runner-18" ]; then
    MULL_VERSION="0.26.1"
    MULL_ZIP="Mull-18-${MULL_VERSION}-LLVM-18.1-macOS-arm64-14.7.4.zip"
    MULL_DIR="Mull-18-${MULL_VERSION}-LLVM-18.1-macOS-arm64-14.7.4"
    curl -LO "https://github.com/mull-project/mull/releases/download/${MULL_VERSION}/${MULL_ZIP}"
    unzip -o "$MULL_ZIP"
    mv "$MULL_DIR" "$BASEDIR/.mull"
    rm "$MULL_ZIP"
  fi
  export PATH="$BASEDIR/.mull/bin:$PATH"
fi

if ! command -v uv &>/dev/null; then
  echo "'uv' is not installed. Installing 'uv'..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # must source this after install on some platforms
  if [ -f $HOME/.local/bin/env ]; then
    source $HOME/.local/bin/env
  fi
fi

export UV_PROJECT_ENVIRONMENT="$BASEDIR/.venv"
uv sync --all-extras --inexact
source "$PYTHONPATH/.venv/bin/activate"

$BASEDIR/opendbc/safety/tests/misra/install.sh
