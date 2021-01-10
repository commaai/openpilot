#!/bin/bash -e

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
    if { ! [ -x "$(command -v pyenv)" ] || ! [ -x "$(command -v make)" ] || ! [ -x "$(command -v curl)" ] || ! [ -x "$(command -v wget)" ]; }; then
        cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null
        if { [ "$DISTRO" == "ubuntu" ] || [ "$DISTRO" == "debian" ] || [ "$DISTRO" == "raspbian" ] || [ "$DISTRO" == "pop" ] || [ "$DISTRO" == "kali" ]; }; then
            apt-get update && apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
        fi
        curl https://pyenv.run | bash
        echo 'export PYENV_ROOT=\"\$HOME/.pyenv\"' >>~/.bashrc
        echo 'export PATH=\"\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH\"' >>~/.bashrc
        exec \"\$SHELL\"
    fi
}

# Run the function and check for requirements
installing-system-requirements

export MAKEFLAGS="-j$(nproc)"

PYENV_PYTHON_VERSION=$(cat .python-version)
if ! pyenv prefix ${PYENV_PYTHON_VERSION} &>/dev/null; then
    echo "pyenv ${PYENV_PYTHON_VERSION} install ..."
    CONFIGURE_OPTS=--enable-shared pyenv install -f ${PYENV_PYTHON_VERSION}
fi

if ! command -v pipenv &>/dev/null; then
    echo "pipenv install ..."
    pip install pipenv
fi

echo "update pip"
pip install --upgrade pip

echo "pip packages install ..."
pipenv install --dev --deploy --system
# update shims for newly installed executables (e.g. scons)
pyenv rehash

echo "precommit install ..."
pre-commit install
