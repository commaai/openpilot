#!/bin/bash -e

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null

if ! command -v pyenv &> /dev/null; then
  echo "please install pyenv ..."
  echo "https://github.com/pyenv/pyenv-installer"
  echo "example:"
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

echo "pip packages install ..."
pipenv install --dev --deploy --system
