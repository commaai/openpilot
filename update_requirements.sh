#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

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
fi
source $RC_FILE

export MAKEFLAGS="-j$(nproc)"

PYENV_PYTHON_VERSION=$(cat .python-version)
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
pip install pip==22.3
pip install poetry==1.2.2

poetry config virtualenvs.prefer-active-python true --local

POETRY_INSTALL_ARGS=""
if [ -d "./xx" ] || [ -n "$XX" ]; then
  echo "WARNING: using xx dependency group, installing globally"
  poetry config virtualenvs.create false --local
  POETRY_INSTALL_ARGS="--with xx --sync"
fi

echo "pip packages install..."
poetry install --no-cache --no-root $POETRY_INSTALL_ARGS
pyenv rehash

if [ -d "./xx" ] || [ -n "$POETRY_VIRTUALENVS_CREATE" ]; then
  RUN=""
else
  echo "PYTHONPATH=${PWD}" > .env
  RUN="poetry run"
fi

echo "pre-commit hooks install..."
shopt -s nullglob
for f in .pre-commit-config.yaml */.pre-commit-config.yaml; do
  cd $DIR/$(dirname $f)
  if [ -e ".git" ]; then
    $RUN pre-commit install
  fi
done
