#!/usr/bin/env bash
set -e

function ask() {
  # Needed to work on Docker setup
  if [[ "$RUN_ALL" == "yes" ]]; then
    return 0
  fi

  read -p "$1 [Y/n] " -n 1 -r
  echo ""
  [[ $REPLY =~ ^[Yy]$ ]] && return 0 || return 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

RC_FILE="${HOME}/.$(basename ${SHELL})rc"
if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
  RC_FILE="$HOME/.bash_profile"
fi

PYENV_INSTALLED = true
if ! command -v "pyenv" > /dev/null 2>&1; then
  PYENV_INSTALLED = false

  if ask "Install pyenv?"; then
    echo "pyenv install ..."
    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
    PYENV_PATH_SETUP="export PATH=\$HOME/.pyenv/bin:\$HOME/.pyenv/shims:\$PATH"

    PYENV_INSTALLED = true
  fi
fi

if $PYENV_INSTALLED && ([ -z "$PYENV_SHELL" ] || [ -n "$PYENV_PATH_SETUP" ]); then
  echo "pyenvrc setup ..."
  cat <<EOF > "${HOME}/.pyenvrc"
if [ -z "\$PYENV_ROOT" ]; then
$PYENV_PATH_SETUP
export PYENV_ROOT="\$HOME/.pyenv"
eval "\$(pyenv init -)"
eval "\$(pyenv virtualenv-init -)"
fi
EOF

  SOURCE_PYENVRC="source ~/.pyenvrc"
  if ! grep "^$SOURCE_PYENVRC$" $RC_FILE > /dev/null; then
    printf "\n$SOURCE_PYENVRC\n" >> $RC_FILE
  fi

  eval "$SOURCE_PYENVRC"
  # $(pyenv init -) produces a function which is broken on bash 3.2 which ships on macOS
  if [ $(uname) == "Darwin" ]; then
    unset -f pyenv
  fi
fi

export MAKEFLAGS="-j$(nproc)"

PYENV_PYTHON_VERSION=$(cat $ROOT/.python-version)
if PYENV_INSTALLED && (! pyenv prefix ${PYENV_PYTHON_VERSION} &> /dev/null); then
  # no pyenv update on mac
  if [ "$(uname)" == "Linux" ]; then
    if ask "Update pyenv?"; then
      echo "pyenv update ..."
      pyenv update
    fi
  fi

  if ask "Install python ${PYENV_PYTHON_VERSION}?"; then
    echo "python ${PYENV_PYTHON_VERSION} install ..."
    CONFIGURE_OPTS="--enable-shared" pyenv install -f ${PYENV_PYTHON_VERSION}
  fi
fi

PYENV_INSTALLED && eval "$(pyenv init --path)"

if ask "Update pip?"; then
  echo "update pip"
  pip install pip==24.0
fi

if ask "Install poetry?"; then
  pip install poetry==1.7.0

  poetry config virtualenvs.prefer-active-python true --local
  poetry config virtualenvs.in-project true --local

  poetry self add poetry-dotenv-plugin@^0.1.0

  POETRY_INSTALLED = true
fi

echo "PYTHONPATH=${PWD}" > $ROOT/.env
if [[ "$(uname)" == 'Darwin' ]]; then
  echo "# msgq doesn't work on mac" >> $ROOT/.env
  echo "export ZMQ=1" >> $ROOT/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> $ROOT/.env

  if [[! POETRY_INSTALLED]]; then
    echo "Before running any python files, load the environment variables by running this command:"
    echo "source ~/.env"
  fi
fi


if POETRY_INSTALLED && PYENV_INSTALLED; then
  if ask "Install openpilot python dependencies?"
    echo "pip packages install..."
    poetry install --no-cache --no-root
    pyenv rehash
  fi
else
  echo "Pyenv or poetry not installed"
  echo "Activate your own virtual environment, then run this command to install the openpilot python dependencies:"
  echo "pip install ."
fi

(! POETRY_INSTALLED || [ -n "$POETRY_VIRTUALENVS_CREATE" ]) && RUN="" || RUN="poetry run"

if [ "$(uname)" != "Darwin" ] && [ -e "$ROOT/.git" ]; then
  if ask "Install git pre-commit hooks?"; then
    echo "pre-commit hooks install..."
    $RUN pre-commit install
    $RUN git submodule foreach pre-commit install
  fi
fi
