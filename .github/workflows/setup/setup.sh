#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
OPENPILOT_ROOT=$DIR/../
cd $ROOT

export INSTALL_EXTRA_PACKAGES=no
export PYTHONUNBUFFERED=1


tools/install_ubuntu_dependencies.sh
tools/install_python_dependencies.sh
