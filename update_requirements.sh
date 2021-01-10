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
    if { ! [ -x "$(command -v pyenv)" ] || ! [ -x "$(command -v make)" ] || ! [ -x "$(command -v curl)" ] || ! [ -x "$(command -v wget)" ]; }; then
        cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null
        if { [ "$DISTRO" == "ubuntu" ] || [ "$DISTRO" == "debian" ] || [ "$DISTRO" == "raspbian" ] || [ "$DISTRO" == "pop" ] || [ "$DISTRO" == "kali" ]; }; then
            apt-get update && apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
        elif { [ "$DISTRO" == "fedora" ] || [ "$DISTRO" == "centos" ] || [ "$DISTRO" == "rhel" ]; }; then
            yum update -y && yum install epel-release iptables curl coreutils bc jq sed e2fsprogs -y
        elif { [ "$DISTRO" == "arch" ] || [ "$DISTRO" == "manjaro" ]; }; then
            pacman -Syu --noconfirm iptables curl bc jq sed
        fi
        curl https://pyenv.run | bash
        echo 'export PYENV_ROOT=\"\$HOME/.pyenv\"' >>~/.bashrc
        echo 'export PATH=\"\$PYENV_ROOT/bin:\$PYENV_ROOT/shims:\$PATH\"' >>~/.bashrc
        exec \"\$SHELL\"
    fi
}

# Run the function and check for requirements
installing-system-requirements

function install-pyenv() {
    export MAKEFLAGS="-j$(nproc)"
    PYENV_PYTHON_VERSION=$(cat .python-version)
    if ! [ -x "$(pyenv prefix ${PYENV_PYTHON_VERSION})" ]; then
        CONFIGURE_OPTS=--enable-shared pyenv install -f ${PYENV_PYTHON_VERSION}
    elif ! [ -x "$(command -v pipenv)" ]; then
        pip install pipenv
    fi
}

install-pyenv

function pip-stuff() {
    pip install --upgrade pip
    pipenv install --dev --deploy --system
    pyenv rehash
    pre-commit install
}

pip-stuff
