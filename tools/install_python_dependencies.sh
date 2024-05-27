#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

RC_FILE="${HOME}/.$(basename ${SHELL})rc"
if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
  RC_FILE="$HOME/.bash_profile"
fi

export MAKEFLAGS="-j$(nproc)"



echo "update pip"
if [ ! -z "\$VIRTUAL_ENV_ROOT" ] || [ ! -z "$INSTALL_DEADSNAKES_PPA" ] ; then
  python3 -m venv --system-site-packages $VIRTUAL_ENV_ROOT
  source $VIRTUAL_ENV_ROOT/bin/activate
fi
pip install pip==24.0
pip install poetry==1.7.0

poetry config virtualenvs.prefer-active-python true --local
poetry config virtualenvs.in-project true --local

echo "PYTHONPATH=${PWD}" > $ROOT/.env
if [[ "$(uname)" == 'Darwin' ]]; then
  echo "# msgq doesn't work on mac" >> $ROOT/.env
  echo "export ZMQ=1" >> $ROOT/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> $ROOT/.env
fi

poetry self add poetry-dotenv-plugin@^0.1.0

echo "pip packages install..."
poetry install --no-cache --no-root

[ -n "$POETRY_VIRTUALENVS_CREATE" ] && RUN="" || RUN="poetry run"

if [ "$(uname)" != "Darwin" ] && [ -e "$ROOT/.git" ]; then
  echo "pre-commit hooks install..."
  $RUN pre-commit install
  $RUN git submodule foreach pre-commit install
fi
