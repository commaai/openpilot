#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

RC_FILE="${HOME}/.$(basename ${SHELL})rc"
if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
  RC_FILE="$HOME/.bash_profile"
fi

if ! command -v "pyenv" > /dev/null 2>&1; then
  echo "pyenv install ..."
  curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

  echo -e "\n. ~/.pyenvrc" >> $RC_FILE
  cat <<EOF > "${HOME}/.pyenvrc"
if [ -z "\$PYENV_ROOT" ]; then
  export PATH=\$HOME/.pyenv/bin:\$HOME/.pyenv/shims:\$PATH
  export PYENV_ROOT="\$HOME/.pyenv"
  eval "\$(pyenv init -)"
  eval "\$(pyenv virtualenv-init -)"
fi
EOF

  # setup now without restarting shell
  export PATH=$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH
  export PYENV_ROOT="$HOME/.pyenv"
  eval "$(pyenv init -)"
  eval "$(pyenv virtualenv-init -)"
fi

export MAKEFLAGS="-j$(nproc)"

PYENV_PYTHON_VERSION=$(cat $ROOT/.python-version)
if ! pyenv prefix ${PYENV_PYTHON_VERSION} &> /dev/null; then
  # no pyenv update on mac
  if [ "$(uname)" == "Linux" ]; then
    echo "pyenv update ..."
    pyenv update
  fi
  echo "python ${PYENV_PYTHON_VERSION} install ..."
  CONFIGURE_OPTS="--enable-shared" pyenv install -f ${PYENV_PYTHON_VERSION}
fi
eval "$(pyenv init --path)"

echo "update pip"
pip install pip==23.2.1
pip install poetry==1.5.1

poetry config virtualenvs.prefer-active-python true --local
poetry config virtualenvs.in-project true --local

echo "PYTHONPATH=${PWD}" > $ROOT/.env
poetry self add poetry-dotenv-plugin@^0.1.0

echo "pip packages install..."
poetry install --no-cache --no-root
pyenv rehash

[ -n "$POETRY_VIRTUALENVS_CREATE" ] && RUN="" || RUN="poetry run"

if [ "$(uname)" != "Darwin" ]; then
  echo "pre-commit hooks install..."
  shopt -s nullglob
  for f in .pre-commit-config.yaml */.pre-commit-config.yaml; do
    if [ -e "$ROOT/$(dirname $f)/.git" ]; then
      $RUN pre-commit install -c "$f"
    fi
  done
fi
