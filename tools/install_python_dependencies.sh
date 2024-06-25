#!/usr/bin/env bash
set -e

# Increase the pip timeout to handle TimeoutError
export PIP_DEFAULT_TIMEOUT=200

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT=$DIR/../
cd $ROOT

function check_python_version() {
  export REQUIRED_PYTHON_VERSION=$(grep "requires-python" pyproject.toml | cut -d= -f3- | tr -d '"' | tr -d ' ')
  if ! command -v "python3" > /dev/null 2>&1; then
    echo "python3 not found on your system. You need python version at least $REQUIRED_PYTHON_VERSION to continue."
    exit 1
  else
    python3 $ROOT/tools/check_python_version.py
  fi
}

function install_precommit() {
  [ -n "$POETRY_VIRTUALENVS_CREATE" ] && RUN="" || RUN="poetry run"

  if [ "$(uname)" != "Darwin" ] && [ -e "$ROOT/.git" ] && [ ! -n "$MANUAL_INSTALL" ]; then
    echo "pre-commit hooks install..."
    $RUN pre-commit install
    $RUN git submodule foreach pre-commit install
  fi
}

function create_dotenv() {
  echo "PYTHONPATH=${PWD}" > $ROOT/.env
  if [[ "$(uname)" == 'Darwin' ]]; then
    echo "# msgq doesn't work on mac" >> $ROOT/.env
    echo "export ZMQ=1" >> $ROOT/.env
    echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> $ROOT/.env
  fi
}

function success_instructions() {
  echo ""
  echo "============ DONE WITH PYTHON DEPENDENCIES ============"

  if [[ -n $MANUAL_INSTALL ]]; then
    echo ""
    echo "To activate the new python virtualenv, run:"
    echo "source $ROOT/.venv/bin/activate"
    echo ""
    echo "To get relevant environment variables, run:"
    echo "source $ROOT/.env"
    echo ""
    echo "Because you did not installed poetry, only runtime python packages were installed!"
  else
    echo ""
    echo "To activate the new python virtualenv, run:"
    echo "source ~/.bashrc"
    echo "poetry shell"
  fi
}

function poetry_installation() {
  RC_FILE="${HOME}/.$(basename ${SHELL})rc"
  if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
    RC_FILE="$HOME/.bash_profile"
  fi

  if ! command -v "poetry" > /dev/null 2>&1; then
    echo "installing poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    POETRY_BIN='$HOME/.local/bin'
    ADD_PATH_CMD="export PATH=\"$POETRY_BIN:\$PATH\""
    eval $ADD_PATH_CMD
    printf "\n#poetry path\n$ADD_PATH_CMD\n" >> $RC_FILE
  fi

  poetry config virtualenvs.prefer-active-python true --local
  poetry config virtualenvs.in-project true --local
  poetry self add poetry-dotenv-plugin@^0.1.0
  poetry install --no-cache --no-root
}

function manual_installation() {
  python3 -m venv .venv --prompt "openpilot_venv"
  source .venv/bin/activate
  pip install .
}

function install_packages() {
  if [[ -z "$INTERACTIVE" ]]; then
    echo ""
    read -p "Do you want to use poetry for managing python packages? (recommended) [Y/n]:" -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      MANUAL_INSTALL=1
      manual_installation
      return 0
    fi
  fi

  poetry_installation
}

check_python_version
install_packages
create_dotenv
install_precommit
success_instructions
