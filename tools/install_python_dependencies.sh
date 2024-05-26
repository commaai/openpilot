#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

export MAKEFLAGS="-j$(nproc)"

echo "update pip"
if [ ! -z "$VIRTUAL_ENV_ROOT" ] || [ ! -z "$INSTALL_DEADSNAKES_PPA" ] ; then
  if [ -z "$VIRTUAL_ENV_ROOT" ]; then
    export VIRTUAL_ENV_ROOT="venv"
  fi
  python3 -m venv --system-site-packages $VIRTUAL_ENV_ROOT
  source $VIRTUAL_ENV_ROOT/bin/activate
fi

echo "PYTHONPATH=${PWD}" > $ROOT/.env
if [[ "$(uname)" == 'Darwin' ]]; then
  echo "# msgq doesn't work on mac" >> $ROOT/.env
  echo "export ZMQ=1" >> $ROOT/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> $ROOT/.env
fi

echo "pip packages install..."

if [ "$(uname)" != "Darwin" ] && [ -e "$ROOT/.git" ]; then
  echo "pre-commit hooks install..."
  $RUN pre-commit install
  $RUN git submodule foreach pre-commit install
fi
