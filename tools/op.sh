#!/usr/bin/env bash

if [[ ! "${BASH_SOURCE[0]}" = "${0}" ]]; then
  echo "Invalid invocation! This script must not be sourced."
  echo "Run 'op.sh' directly or check your .bashrc for a valid alias"
  return 0
fi

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
UNDERLINE='\033[4m'
BOLD='\033[1m'
NC='\033[0m'

RC_FILE="${HOME}/.$(basename ${SHELL})rc"
if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
  RC_FILE="$HOME/.bash_profile"
fi
function op_install() {
  echo "Installing op system-wide..."
  CMD="\nalias op='"$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/op.sh" \"\$@\"'\n"
  grep "alias op=" "$RC_FILE" &> /dev/null || printf "$CMD" >> $RC_FILE
  echo -e " ↳ [${GREEN}✔${NC}] op installed successfully. Open a new shell to use it.\n"
}

function op_run_command() {
  CMD="$@"
  echo -e "${BOLD}Running:${NC} $CMD"
  if [[ -z "$DRY" ]]; then
    eval "$CMD"
  fi
}

# be default, assume openpilot dir is in current directory
OPENPILOT_ROOT=$(pwd)
function op_get_openpilot_dir() {
  while [[ "$OPENPILOT_ROOT" != '/' ]];
  do
    if find "$OPENPILOT_ROOT/launch_openpilot.sh" -maxdepth 1 -mindepth 1 &> /dev/null; then
      return 0
    fi
    OPENPILOT_ROOT="$(readlink -f "$OPENPILOT_ROOT/"..)"
  done
}

function op_check_openpilot_dir() {
  echo "Checking for openpilot directory..."
  if [[ -f "$OPENPILOT_ROOT/launch_openpilot.sh" ]]; then
    echo -e " ↳ [${GREEN}✔${NC}] openpilot found.\n"
    return 0
  fi

  echo -e " ↳ [${RED}✗${NC}] openpilot directory not found! Make sure that you are"
  echo "       inside the openpilot directory or specify one with the"
  echo "       --dir option!"
  return 1
}

function op_check_git() {
  echo "Checking for git..."
  if ! command -v "git" > /dev/null 2>&1; then
    echo -e " ↳ [${RED}✗${NC}] git not found on your system!"
    return 1
  else
    echo -e " ↳ [${GREEN}✔${NC}] git found."
  fi

  echo "Checking for git lfs files..."
  if [[ $(file -b $OPENPILOT_ROOT/selfdrive/modeld/models/supercombo.onnx) == "data" ]]; then
    echo -e " ↳ [${GREEN}✔${NC}] git lfs files found."
  else
    echo -e " ↳ [${RED}✗${NC}] git lfs files not found! Run 'git lfs pull'"
    return 1
  fi

  echo "Checking for git submodules..."
  for name in $(git config --file .gitmodules --get-regexp path | awk '{ print $2 }' | tr '\n' ' '); do
    if [[ -z $(ls $OPENPILOT_ROOT/$name) ]]; then
      echo -e " ↳ [${RED}✗${NC}] git submodule $name not found! Run 'git submodule update --init --recursive'"
      return 1
    fi
  done
  echo -e " ↳ [${GREEN}✔${NC}] git submodules found."
}

function op_check_os() {
  echo "Checking for compatible os version..."
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then

    if [ -f "/etc/os-release" ]; then
      source /etc/os-release
      case "$VERSION_CODENAME" in
        "jammy" | "kinetic" | "noble" | "focal")
          echo -e " ↳ [${GREEN}✔${NC}] Ubuntu $VERSION_CODENAME detected."
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
}

function op_check_python() {
  echo "Checking for compatible python version..."
  REQUIRED_PYTHON_VERSION=$(grep "requires-python" $OPENPILOT_ROOT/pyproject.toml)
  INSTALLED_PYTHON_VERSION=$(python3 --version 2> /dev/null || true)

  if [[ -z $INSTALLED_PYTHON_VERSION ]]; then
    echo -e " ↳ [${RED}✗${NC}] python3 not found on your system. You need python version at least $(echo $REQUIRED_PYTHON_VERSION | tr -d -c '[0-9.]') to continue!"
    return 1
  elif [[ $(echo $INSTALLED_PYTHON_VERSION | grep -o '[0-9]\+\.[0-9]\+' | tr -d -c '[0-9]') -ge $(echo $REQUIRED_PYTHON_VERSION | tr -d -c '[0-9]') ]]; then
    echo -e " ↳ [${GREEN}✔${NC}] $INSTALLED_PYTHON_VERSION detected."
  else
    echo -e " ↳ [${RED}✗${NC}] You need python version at least $(echo $REQUIRED_PYTHON_VERSION | tr -d -c '[0-9.]') to continue!"
    return 1
  fi
}

function op_check_venv() {
  echo "Checking for venv..."
  if source $OPENPILOT_ROOT/.venv/bin/activate; then
    echo -e " ↳ [${GREEN}✔${NC}] venv detected."
  else
    echo -e " ↳ [${RED}✗${NC}] Can't activate venv in $OPENPILOT_ROOT. Assuming global env!"
  fi
}

function op_before_cmd() {
  if [[ ! -z "$NO_VERIFY" ]]; then
    return 0
  fi

  op_get_openpilot_dir
  cd $OPENPILOT_ROOT

  result="$((op_check_openpilot_dir ) 2>&1)" || (echo -e "$result" && return 1)
  result="${result}\n$(( op_check_git ) 2>&1)" || (echo -e "$result" && return 1)
  result="${result}\n$(( op_check_os ) 2>&1)" || (echo -e "$result" && return 1)
  result="${result}\n$(( op_check_venv ) 2>&1)" || (echo -e "$result" && return 1)

  op_activate_venv

  result="${result}\n$(( op_check_python ) 2>&1)" || (echo -e "$result" && return 1)

  if [[ -z $VERBOSE ]]; then
    echo -e "Checking system → [${GREEN}✔${NC}] system is good."
  else
    echo -e "$result"
  fi
}

function op_setup() {
  op_get_openpilot_dir
  cd $OPENPILOT_ROOT

  op_check_openpilot_dir
  op_check_os
  op_check_python

  echo "Installing dependencies..."
  st="$(date +%s)"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    op_run_command $OPENPILOT_ROOT/tools/ubuntu_setup.sh
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    op_run_command $OPENPILOT_ROOT/tools/mac_setup.sh
  fi
  et="$(date +%s)"
  echo -e " ↳ [${GREEN}✔${NC}] Dependencies installed successfully in $((et - st)) seconds.\n"

  echo "Getting git submodules..."
  st="$(date +%s)"
  op_run_command git submodule update --filter=blob:none --jobs 4 --init --recursive
  et="$(date +%s)"
  echo -e " ↳ [${GREEN}✔${NC}] Submodules installed successfully in $((et - st)) seconds.\n"

  echo "Pulling git lfs files..."
  st="$(date +%s)"
  op_run_command git lfs pull
  et="$(date +%s)"
  echo -e " ↳ [${GREEN}✔${NC}] Files pulled successfully in $((et - st)) seconds.\n"

  op_check
}

function op_activate_venv() {
  source $OPENPILOT_ROOT/.venv/bin/activate &> /dev/null || true
}

function op_venv() {
  op_before_cmd
  bash --rcfile <(echo "source $RC_FILE; source $OPENPILOT_ROOT/.venv/bin/activate")
}

function op_check() {
  VERBOSE=1
  op_before_cmd
  unset VERBOSE
}

function op_build() {
  CDIR=$(pwd)
  op_before_cmd
  cd "$CDIR"
  op_run_command scons $@
}

function op_juggle() {
  op_before_cmd
  op_run_command tools/plotjuggler/juggle.py $@
}

function op_lint() {
  op_before_cmd
  op_run_command scripts/lint.sh $@
}

function op_test() {
  op_before_cmd
  op_run_command pytest $@
}

function op_replay() {
  op_before_cmd
  op_run_command tools/replay/replay $@
}

function op_cabana() {
  op_before_cmd
  op_run_command tools/cabana/cabana $@
}

function op_sim() {
  op_before_cmd
  op_run_command exec tools/sim/run_bridge.py &
  op_run_command exec tools/sim/launch_openpilot.sh
}

function op_default() {
  echo "An openpilot helper"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Description:${NC}"
  echo "  op is your entry point for all things related to openpilot development."
  echo "  op is only a wrapper for existing scripts, tools, and commands."
  echo "  op will always show you what it will run on your system."
  echo ""
  echo "  op will try to find your openpilot directory in the following order:"
  echo "   1: use the directory specified with the --dir option"
  echo "   2: use the current working directory"
  echo "   3: go up the file tree non-recursively"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Usage:${NC} op [OPTIONS] <COMMAND>"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Commands:${NC}"
  echo -e "  ${BOLD}venv${NC}     Activate the Python virtual environment"
  echo -e "  ${BOLD}check${NC}    Check the development environment (git, os, python) to start using openpilot"
  echo -e "  ${BOLD}setup${NC}    Install openpilot dependencies"
  echo -e "  ${BOLD}build${NC}    Run the openpilot build system in the current working directory"
  echo -e "  ${BOLD}sim${NC}      Run openpilot in a simulator"
  echo -e "  ${BOLD}juggle${NC}   Run Plotjuggler"
  echo -e "  ${BOLD}replay${NC}   Run replay"
  echo -e "  ${BOLD}cabana${NC}   Run cabana"
  echo -e "  ${BOLD}lint${NC}     Run the linter"
  echo -e "  ${BOLD}test${NC}     Run all unit tests from pytest"
  echo -e "  ${BOLD}help${NC}     Show this message"
  echo -e "  ${BOLD}install${NC}  Install the 'op' tool system wide"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Options:${NC}"
  echo -e "  ${BOLD}-d, --dir${NC}"
  echo "          Specify the openpilot directory you want to use"
  echo -e "  ${BOLD}--dry${NC}"
  echo "          Don't actually run anything, just print what would be run"
  echo -e "  ${BOLD}-n, --no-verify${NC}"
  echo "          Skip environment check before running commands"
  echo -e "  ${BOLD}-v, --verbose${NC}"
  echo "          Show the result of all checks before running a command"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Examples:${NC}"
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
    -d | --dir )       shift 1; OPENPILOT_ROOT="$1"; shift 1 ;;
    --dry )            shift 1; DRY="1" ;;
    -n | --no-verify ) shift 1; NO_VERIFY="1" ;;
    -v | --verbose )   shift 1; VERBOSE="1" ;;
  esac

  # parse Commands
  case $1 in
    venv )      shift 1; op_venv "$@" ;;
    check )     shift 1; op_check "$@" ;;
    setup )     shift 1; op_setup "$@" ;;
    build )     shift 1; op_build "$@" ;;
    juggle )    shift 1; op_juggle "$@" ;;
    cabana )    shift 1; op_cabana "$@" ;;
    lint )      shift 1; op_lint "$@" ;;
    test )      shift 1; op_test "$@" ;;
    replay )    shift 1; op_replay "$@" ;;
    sim )       shift 1; op_sim "$@" ;;
    install )   shift 1; op_install "$@" ;;
    * ) op_default "$@" ;;
  esac
}

_op $@
