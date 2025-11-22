#!/bin/bash
set -e

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# TODO: why doesn't uv do this?
export PYTHONPATH=$BASEDIR

# *** dependencies install ***
if [ "$(uname -s)" = "Linux" ]; then
  # TODO: add macOS support
  if ! command -v "mull-runner-17" > /dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y curl clang-17
    curl -1sLf 'https://dl.cloudsmith.io/public/mull-project/mull-stable/setup.deb.sh' | sudo -E bash
    sudo apt-get update && sudo apt-get install -y mull-17
  fi
fi

if ! command -v uv &>/dev/null; then
  echo "'uv' is not installed. Installing 'uv'..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export UV_PROJECT_ENVIRONMENT="$BASEDIR/.venv"
uv sync --all-extras
source "$PYTHONPATH/.venv/bin/activate"

$BASEDIR/opendbc/safety/tests/misra/install.sh
