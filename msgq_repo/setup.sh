#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

PLATFORM=$(uname -s)

# catch2
if [ ! -d $DIR/msgq/catch2/ ]; then
  rm -rf /tmp/catch2/ $DIR/msgq/catch2/
  git clone -b v2.x --depth 1 https://github.com/catchorg/Catch2.git /tmp/catch2
  pushd /tmp/catch2
  mv single_include/* $DIR/msgq/
  popd
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
