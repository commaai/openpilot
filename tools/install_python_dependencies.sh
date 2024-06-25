#!/usr/bin/env bash
set -e

# Increase the pip timeout to handle TimeoutError
export PIP_DEFAULT_TIMEOUT=200

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

if ! command -v "uv" > /dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN='$HOME/.cargo/env'
  ADD_PATH_CMD=". \"$UV_BIN\""
  eval $ADD_PATH_CMD
fi

if [ -z "$VIRTUAL_ENV" ]; then
  echo "creating virtual env..."
  uv venv
  . .venv/bin/activate

  RC_FILE="${HOME}/.$(basename ${SHELL})rc"
  if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
    RC_FILE="$HOME/.bash_profile"
  fi

  printf "\n#openpilot venv\nalias openpilot_shell=\". $VIRTUAL_ENV/bin/activate && . $ROOT/.env \"\n" >> $RC_FILE
fi

echo "installing python packages..."
uv pip sync requirements.txt

echo "PYTHONPATH=${PWD}" > $ROOT/.env
if [[ "$(uname)" == 'Darwin' ]]; then
  echo "# msgq doesn't work on mac" >> $ROOT/.env
  echo "export ZMQ=1" >> $ROOT/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> $ROOT/.env
fi

if [ "$(uname)" != "Darwin" ] && [ -e "$ROOT/.git" ]; then
  echo "pre-commit hooks install..."
  pre-commit install
  git submodule foreach pre-commit install
fi
