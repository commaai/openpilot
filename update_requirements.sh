#!/bin/bash -e

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null

if ! command -v pyenv &> /dev/null; then
  echo "please install pyenv ..."
  echo "https://github.com/pyenv/pyenv-installer"
  echo "example:"
  echo "sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev"
  echo "curl https://pyenv.run | bash"
  echo "echo 'export PYENV_ROOT=\"\$HOME/.pyenv\"' >> ~/.bashrc"
  echo "echo 'export PATH=\"\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH\"' >> ~/.bashrc"
  echo "exec \"\$SHELL\""
  exit 1
fi

export MAKEFLAGS="-j$(nproc)"

PYENV_PYTHON_VERSION=$(cat .python-version)
if ! pyenv prefix ${PYENV_PYTHON_VERSION} &> /dev/null; then
  echo "pyenv ${PYENV_PYTHON_VERSION} install ..."
  CONFIGURE_OPTS=--enable-shared pyenv install -f ${PYENV_PYTHON_VERSION}
fi

if ! command -v pipenv &> /dev/null; then
  echo "pipenv install ..."
  pip install pipenv
fi

echo "update pip"
pip install pip==21.3.1
pip install pipenv==2021.5.29

echo "pip packages install ..."
if [ -d "./xx" ]; then
  PIPENV_PIPFILE=./xx/Pipfile pipenv install --system --dev --deploy
  pipenv install --system --dev --deploy
  RUN=""
else
  pipenv install --dev --deploy
  RUN="pipenv run"
fi

# update shims for newly installed executables (e.g. scons)
pyenv rehash

echo "precommit install ..."
$RUN pre-commit install

# for internal comma repos
[ -d "./xx" ] && (cd xx && $RUN pre-commit install)
[ -d "./notebooks" ] && (cd notebooks && $RUN pre-commit install)
