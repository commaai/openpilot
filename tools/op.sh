#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

function op_first_install() {
  (set -e

  echo "Installing op system-wide..."
  RC_FILE="${HOME}/.$(basename ${SHELL})rc"
  if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
    RC_FILE="$HOME/.bash_profile"
  fi
  printf "\nalias op='source "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/op.sh" \"\$@\"'\n" >> $RC_FILE
  echo -e " ↳ [${GREEN}✔${NC}] op installed successfully. Open a new shell to use it.\n"

  )
}

# be default, assume openpilot dir is in current directory
OPENPILOT_ROOT=$(pwd)
function op_check_openpilot_dir() {
  (set -e

  if [ ! -f "$OPENPILOT_ROOT/launch_openpilot.sh" ]; then
    echo "openpilot directory not found!"
    return 1
  fi

  )
}

function op_check_git() {
  (set -e

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

  echo "Checking for git submodules..."
  if $(git submodule foreach --quiet --recursive 'return 1' 2&> /dev/null); then
    echo -e " ↳ [${RED}✗${NC}] git submodules not found! Run 'git submodule update --init --recursive'"
    return 1
  else
    echo -e " ↳ [${GREEN}✔${NC}] git submodules found on your system.\n"
  fi

  )
}

function op_check_os() {
  (set -e

  echo "Checking for compatible os version..."
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then

    if [ -f "/etc/os-release" ]; then
      source /etc/os-release
      case "$VERSION_CODENAME" in
        "jammy" | "kinetic" | "noble" | "focal")
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
  else
    echo -e " ↳ [${RED}✗${NC}] OS type $OSTYPE not supported!"
    return 1
  fi

  )
}

function op_check_python() {
  (set -e

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

  )
}

# this must be run in the same shell as the user calling "op"
function op_venv() {
  op_check_openpilot_dir || return 1
  source $OPENPILOT_ROOT/.venv/bin/activate || (echo -e "\nCan't activate venv. Have you ran 'op install' ?" && return 1)
}

function op_check() {
  (set -e

  op_check_openpilot_dir
  cd $OPENPILOT_ROOT
  op_check_git
  op_check_os
  op_check_python

  )
}

function op_run() {
  (set -e

  op_venv
  cd $OPENPILOT_ROOT
  $OPENPILOT_ROOT/launch_openpilot.sh

  )
}

function op_install() {
  (set -e

  op_check_openpilot_dir
  cd $OPENPILOT_ROOT

  op_check_os
  op_check_python

  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    $OPENPILOT_ROOT/tools/ubuntu_setup.sh
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    $OPENPILOT_ROOT/tools/mac_setup.sh
  fi

  git submodule update --init --recursive
  git lfs pull

  )
}

function op_build() {
  (set -e

  op_venv
  cd $OPENPILOT_ROOT

  scons $@ || echo -e "\nBuild failed. Have you ran 'op install' ?"

  )
}

function op_juggle() {
  (set -e

  op_venv
  cd $OPENPILOT_ROOT

  $OPENPILOT_ROOT/tools/plotjuggler/juggle.py $@

  )
}

function op_default() {
  echo "An openpilot helper"
  echo ""
  echo -e "\e[4mUsage:\e[0m op [OPTIONS] <COMMAND>"
  echo ""
  echo -e "\e[4mCommands:\e[0m"
  echo "  venv       Activate the virtual environment"
  echo "  check      Check system requirements (git, os, python) to start using openpilot"
  echo "  install    Install requirements to use openpilot"
  echo "  build      Build openpilot"
  echo "  run        Run openpilot"
  echo "  juggle     Run Plotjuggler"
  echo "  help       Show this message"
  echo "  --install  Install this tool system wide"
  echo ""
  echo -e "\e[4mOptions:\e[0m"
  echo "  -d, --dir"
  echo "          Specify the openpilot directory you want to use"
  echo "          Default to the current working directory"
  echo ""
  echo -e "\e[4mExamples:\e[0m"
  echo "  op --dir /tmp/openpilot check"
  echo "          Run the check command on openpilot located in /tmp/openpilot"
  echo ""
  echo "  op juggle --install"
  echo "          Install plotjuggler in the openpilot located in your current"
  echo "          working directory"
  echo ""
  echo "  op --dir /tmp/openpilot build -j4"
  echo "          Run the build command on openpilot located in /tmp/openpilot"
  echo "          on 4 cores"
}


function _op() {
  # parse Options
  case $1 in
    -d | --dir ) shift 1; OPENPILOT_ROOT="$1"; shift 1 ;;
  esac

  # parse Commands
  case $1 in
    venv )      shift 1; op_venv "$@" ;;
    check )     shift 1; op_check "$@" ;;
    install )   shift 1; op_install "$@" ;;
    build )     shift 1; op_build "$@" ;;
    run )       shift 1; op_run "$@" ;;
    juggle )    shift 1; op_juggle "$@" ;;
    --install ) shift 1; op_first_install "$@" ;;
    * ) op_default "$@" ;;
  esac
}

_op $@

# remove from env
unset -f _op
unset -f op_check
unset -f op_install
unset -f op_build
unset -f op_run
unset -f op_juggle
unset -f op_venv
unset -f op_check_openpilot_dir
unset -f op_check_git
unset -f op_check_python
unset -f op_check_os
unset -f op_first_install
unset -f op_default
