#!/usr/bin/env bash
set -e

# Increase the pip timeout to handle TimeoutError
export PIP_DEFAULT_TIMEOUT=200

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR"/../
cd "$ROOT"

if ! command -v "uv" > /dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
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
  echo "# msgq doesn't work on mac" >> "$ROOT"/.env
  echo "export ZMQ=1" >> "$ROOT"/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> "$ROOT"/.env
fi
