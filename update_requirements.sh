#!/bin/bash

# Require script to be run as root
function super-user-check() {
  if [ "$EUID" -ne 0 ]; then
    echo "You need to run this script as super user."
    exit
  fi
}

# Check for root
super-user-check

# Detect Operating System
function dist-check() {
  if [ -e /etc/os-release ]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    DISTRO=$ID
    DISTRO_VERSION=$VERSION_ID
  fi
}

# Check Operating System
dist-check

# Pre-Checks
function installing-system-requirements() {
  if [ "$DISTRO" == "ubuntu" ] && [ "$DISTRO_VERSION" == "20.04" ]; then
    if { ! [ -x "$(command -v pyenv)" ] || ! [ -x "$(command -v make)" ] || ! [ -x "$(command -v curl)" ] || ! [ -x "$(command -v wget)" ]; }; then
      apt-get update && apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
      curl https://pyenv.run | bash
      echo "export PYENV_ROOT=""$HOME":/.pyenv"" >>~/.bashrc
      echo "export PATH=""$PYENV_ROOT":/"$PYENV_ROOT"/shims/"$PATH""" >>~/.bashrc
      exec "$SHELL"
    fi
  else
    exit
  fi
}

# Run the function and check for requirements
installing-system-requirements

# Install pyenv
function install-pyenv() {
  if [ "$DISTRO" == "ubuntu" ] && [ "$DISTRO_VERSION" == "20.04" ]; then
    export MAKEFLAGS="-j$(nproc)"
    cd "$(dirname "${BASH_SOURCE[0]}")" || exit
    PYENV_PYTHON_VERSION=$(cat .python-version)
    if ! [ -x "$(pyenv prefix "${PYENV_PYTHON_VERSION}")" ]; then
      CONFIGURE_OPTS=--enable-shared pyenv install -f "${PYENV_PYTHON_VERSION}"
    elif ! [ -x "$(command -v pipenv)" ]; then
      pip install pipenv
    fi
  else
    exit
  fi
}

# Install pyenv
install-pyenv

# Update Pip
function pip-stuff() {
  if [ "$DISTRO" == "ubuntu" ] && [ "$DISTRO_VERSION" == "20.04" ]; then
    if [ -x "$(command -v pip)" ]; then
      pip install --upgrade pip
    elif [ -x "$(command -v pipenv)" ]; then
      pipenv install --dev --deploy --system
    elif [ -x "$(command -v pyenv)" ]; then
      pyenv rehash
      pre-commit install
    fi
  else
    exit
  fi
}

# Update Pip, && pipenv
pip-stuff
