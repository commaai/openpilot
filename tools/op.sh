#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# be default, assume openpilot dir is in current directory
OPENPILOT_ROOT=$(pwd)
function op_check_openpilot_dir() {
  if [ ! -f "$OPENPILOT_ROOT/launch_openpilot.sh" ]; then
    echo "openpilot directory not found!"
    return 1
  fi
}

function op_check_git() {
  cd $OPENPILOT_ROOT

  echo "Checking for git..."
  if ! command -v "git" > /dev/null 2>&1; then
    echo -e " ↳ [${RED}✗${NC}] git not found on your system!"
    return 1
  else
    echo -e " ↳ [${GREEN}✔${NC}] git found on your system.\n"
  fi

  echo "Checking for git lfs files..."
  if [[ $(file -b $(git lfs ls-files -n | grep "\.so" | head -n 1)) == "ASCII text" ]]; then
    echo -e " ↳ [${RED}✗${NC}] git lfs files not found! Run git lfs pull"
    return 1
  else
    echo -e " ↳ [${GREEN}✔${NC}] git lfs files found on your system.\n"
  fi
}

function op_check_os() {
  echo "Checking for compatible os version..."
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then

    if [ -f "/etc/os-release" ]; then
      source /etc/os-release
      case "$VERSION_CODENAME" in
        "jammy" | "kinetic" | "noble" | "focal")
          OS_VERSION="Ubuntu"
          echo -e " ↳ [${GREEN}✔${NC}] Ubuntu $VERSION_CODENAME detected.\n"
          ;;
        * )
          echo -e " ↳ [${RED}✗${NC}] Incompatible Ubuntu version $VERSION_CODENAME detected!"
          return 1
          ;;
      esac
    else
      echo -e " ↳ [${RED}✗${NC}] No /etc/os-release on your system. Make sure you're running on Ubuntu, or similar!"
      return 1
    fi

  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e " ↳ [${GREEN}✔${NC}] macos detected.\n"
    OS_VERSION="Darwin"
  else
    echo -e " ↳ [${RED}✗${NC}] OS type $OSTYPE not supported!"
    return 1
  fi
}

function op_check_python() {
  echo "Checking for compatible python version..."
  export REQUIRED_PYTHON_VERSION=$(grep "requires-python" pyproject.toml | cut -d= -f3- | tr -d '"' | tr -d ' ')
  if ! command -v "python3" > /dev/null 2>&1; then
    echo -e " ↳ [${RED}✗${NC}] python3 not found on your system. You need python version at least $REQUIRED_PYTHON_VERSION to continue!"
    return 1
  else
    if $(python3 -c "import sys; quit(not sys.version_info >= tuple(map(int, \"$REQUIRED_PYTHON_VERSION\".split('.'))))"); then
      echo -e " ↳ [${GREEN}✔${NC}] $(python3 --version) detected.\n"
    else
      echo -e " ↳ [${RED}✗${NC}] You need python version at least $REQUIRED_PYTHON_VERSION to continue!"
      return 1
    fi
  fi
}

function op_venv() {
  op_check_openpilot_dir
  . $OPENPILOT_ROOT/.venv/bin/activate 2&> /dev/null || (echo "Can't activate venv. Have you ran 'op install' ?" && return 1)
}

function op_check() {
  op_check_openpilot_dir
  cd $OPENPILOT_ROOT
  op_check_git
  op_check_os
  op_check_python
}

function op_run() {
  op_venv
  cd $OPENPILOT_ROOT
  $OPENPILOT_ROOT/launch_openpilot.sh
}

function op_install() {
  op_check_openpilot_dir
  cd $OPENPILOT_ROOT

  op_check_os
  op_check_python

  case "$OS_VERSION" in
    "Ubuntu")
      $OPENPILOT_ROOT/tools/ubuntu_setup.sh
      ;;
    "Darwin")
      $OPENPILOT_ROOT/tools/mac_setup.sh
      ;;
  esac

  git lfs pull
}

function op_build() {
  op_venv
  cd $OPENPILOT_ROOT

  scons -j$(nproc || sysctl -n hw.logicalcpu)
}

function op_juggle() {
  op_venv
  cd $OPENPILOT_ROOT

  $OPENPILOT_ROOT/tools/plotjuggler/juggle.py
}

function op_default() {
  echo "An openpilot helper"
  echo ""
  echo -e "\e[4mUsage:\e[0m op <COMMAND>"
  echo ""
  echo -e "\e[4mCommands:\e[0m"
  echo "  check    Check system requirements (git, os, python) to start using openpilot"
  echo "  install  Install requirements to use openpilot"
  echo "  build    Build openpilot"
  echo "  run      Run openpilot"
  echo "  juggle   Plotjuggler"
  echo "  linter   Run the linter"
  echo "  help     Show this message"
}

function op() {
  case $1 in
    check )  shift 1; op_check "$@" ;;
    install )    shift 1; op_install "$@" ;;
    build )    shift 1; op_build "$@" ;;
    run )    shift 1; op_run "$@" ;;
    juggle ) shift 1; op_juggle "$@" ;;
    * ) op_default "$@" ;;
  esac
}

op $@
