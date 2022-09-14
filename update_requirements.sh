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
pip install pip==21.3.1
pip install pipenv==2021.11.23

if [ -d "./xx" ]; then
  echo "WARNING: using xx Pipfile ******"
  export PIPENV_SYSTEM=1
  export PIPENV_PIPFILE=./xx/Pipfile
fi

if [ -z "$PIPENV_SYSTEM" ]; then
  echo "PYTHONPATH=${PWD}" > .env
  RUN="pipenv run"
else
  RUN=""
fi

echo "pip packages install..."
pipenv sync --dev
pipenv --clear
pyenv rehash

echo "pre-commit hooks install..."
shopt -s nullglob
for f in .pre-commit-config.yaml */.pre-commit-config.yaml; do
  cd $DIR/$(dirname $f)
  if [ -e ".git" ]; then
    $RUN pre-commit install
  fi
done
