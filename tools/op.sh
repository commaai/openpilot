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

SHELL_NAME="$(basename ${SHELL})"
RC_FILE="${HOME}/.$(basename ${SHELL})rc"
if [ "$(uname)" == "Darwin" ] && [ $SHELL == "/bin/bash" ]; then
  RC_FILE="$HOME/.bash_profile"
fi
function op_install() {
  echo "Installing op system-wide..."
  CMD="\nalias op='"$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/op.sh" \"\$@\"'\n"
  grep "alias op=" "$RC_FILE" &> /dev/null || printf "$CMD" >> $RC_FILE
  echo -e " ↳ [${GREEN}✔${NC}] op installed successfully. Open a new shell to use it."
}

function loge() {
  if [[ -f "$LOG_FILE" ]]; then
    # error type
    echo "$1" >> $LOG_FILE
    # error log
    echo "$2" >> $LOG_FILE
  fi
}

function op_run_command() {
  CMD="$@"

  echo -e "${BOLD}Running command →${NC} $CMD │"
  for ((i=0; i<$((19 + ${#CMD})); i++)); do
    echo -n "─"
  done
  echo -e "┘\n"

  if [[ -z "$DRY" ]]; then
    eval "$CMD"
  fi
}

# be default, assume openpilot dir is in current directory
OPENPILOT_ROOT=$(pwd)
function op_get_openpilot_dir() {
  # First try traversing up the directory tree
  while [[ "$OPENPILOT_ROOT" != '/' ]];
  do
    if find "$OPENPILOT_ROOT/launch_openpilot.sh" -maxdepth 1 -mindepth 1 &> /dev/null; then
      return 0
    fi
    OPENPILOT_ROOT="$(readlink -f "$OPENPILOT_ROOT/"..)"
  done

  # Fallback to hardcoded directories if not found
  for dir in "$HOME/openpilot" "/data/openpilot"; do
    if [[ -f "$dir/launch_openpilot.sh" ]]; then
      OPENPILOT_ROOT="$dir"
      return 0
    fi
  done
}

function op_install_post_commit() {
  op_get_openpilot_dir
  if [[ ! -d $OPENPILOT_ROOT/.git/hooks/post-commit.d ]]; then
    mkdir $OPENPILOT_ROOT/.git/hooks/post-commit.d
    mv $OPENPILOT_ROOT/.git/hooks/post-commit $OPENPILOT_ROOT/.git/hooks/post-commit.d 2>/dev/null || true
  fi
  cd $OPENPILOT_ROOT/.git/hooks
  ln -sf ../../scripts/post-commit post-commit
}

function op_check_openpilot_dir() {
  echo "Checking for openpilot directory..."
  if [[ -f "$OPENPILOT_ROOT/launch_openpilot.sh" ]]; then
    echo -e " ↳ [${GREEN}✔${NC}] openpilot found."
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
  if [[ $(file -b $OPENPILOT_ROOT/selfdrive/modeld/models/dmonitoring_model.onnx) == "data" ]]; then
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
          loge "ERROR_INCOMPATIBLE_UBUNTU" "$VERSION_CODENAME"
          return 1
          ;;
      esac
    else
      echo -e " ↳ [${RED}✗${NC}] No /etc/os-release on your system. Make sure you're running on Ubuntu, or similar!"
      loge "ERROR_UNKNOWN_UBUNTU"
      return 1
    fi

  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e " ↳ [${GREEN}✔${NC}] macOS detected."
  else
    echo -e " ↳ [${RED}✗${NC}] OS type $OSTYPE not supported!"
    loge "ERROR_UNKNOWN_OS" "$OSTYPE"
    return 1
  fi
}

function op_check_python() {
  echo "Checking for compatible python version..."
  REQUIRED_PYTHON_VERSION=$(grep "requires-python" $OPENPILOT_ROOT/pyproject.toml)
  INSTALLED_PYTHON_VERSION=$(python3 --version 2> /dev/null || true)

  if [[ -z $INSTALLED_PYTHON_VERSION ]]; then
    echo -e " ↳ [${RED}✗${NC}] python3 not found on your system. You need python version satisfying $(echo $REQUIRED_PYTHON_VERSION | cut -d '=' -f2-) to continue!"
    loge "ERROR_PYTHON_NOT_FOUND"
    return 1
  else
    LB=$(echo $REQUIRED_PYTHON_VERSION | tr -d -c '[0-9,]' | cut -d ',' -f1)
    UB=$(echo $REQUIRED_PYTHON_VERSION | tr -d -c '[0-9,]' | cut -d ',' -f2)
    VERSION=$(echo $INSTALLED_PYTHON_VERSION | grep -o '[0-9]\+\.[0-9]\+' | tr -d -c '[0-9]')
    if [[ $VERSION -ge LB && $VERSION -lt UB ]]; then
      echo -e " ↳ [${GREEN}✔${NC}] $INSTALLED_PYTHON_VERSION detected."
    else
      echo -e " ↳ [${RED}✗${NC}] You need a python version satisfying $(echo $REQUIRED_PYTHON_VERSION | cut -d '=' -f2-) to continue!"
      loge "ERROR_PYTHON_VERSION" "$INSTALLED_PYTHON_VERSION"
      return 1
    fi
  fi
}

function op_check_venv() {
  echo "Checking for venv..."
  if [[ -f $OPENPILOT_ROOT/.venv/bin/activate ]]; then
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
    echo -e "${BOLD}Checking system →${NC} [${GREEN}✔${NC}]"
  else
    echo -e "$result"
  fi
}

function op_setup() {
  op_get_openpilot_dir
  cd $OPENPILOT_ROOT

  op_check_openpilot_dir
  op_check_os

  echo "Installing dependencies..."
  st="$(date +%s)"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SETUP_SCRIPT="tools/ubuntu_setup.sh"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    SETUP_SCRIPT="tools/mac_setup.sh"
  fi
  if ! $OPENPILOT_ROOT/$SETUP_SCRIPT; then
    echo -e " ↳ [${RED}✗${NC}] Dependencies installation failed!"
    loge "ERROR_DEPENDENCIES_INSTALLATION"
    return 1
  fi
  et="$(date +%s)"
  echo -e " ↳ [${GREEN}✔${NC}] Dependencies installed successfully in $((et - st)) seconds."

  echo "Getting git submodules..."
  st="$(date +%s)"
  if ! git submodule update --filter=blob:none --jobs 4 --init --recursive; then
    echo -e " ↳ [${RED}✗${NC}] Getting git submodules failed!"
    loge "ERROR_GIT_SUBMODULES"
    return 1
  fi
  et="$(date +%s)"
  echo -e " ↳ [${GREEN}✔${NC}] Submodules installed successfully in $((et - st)) seconds."

  echo "Pulling git lfs files..."
  st="$(date +%s)"
  if ! git lfs pull; then
    echo -e " ↳ [${RED}✗${NC}] Pulling git lfs files failed!"
    loge "ERROR_GIT_LFS"
    return 1
  fi
  et="$(date +%s)"
  echo -e " ↳ [${GREEN}✔${NC}] Files pulled successfully in $((et - st)) seconds."

  op_check
}

function op_auth() {
  op_before_cmd
  op_run_command tools/lib/auth.py
}

function op_activate_venv() {
  # bash 3.2 can't handle this without the 'set +e'
  set +e
  source $OPENPILOT_ROOT/.venv/bin/activate &> /dev/null || true
  set -e
}

function op_venv() {
  op_before_cmd

  if [[ ! -f $OPENPILOT_ROOT/.venv/bin/activate ]]; then
    echo -e "No venv found in $OPENPILOT_ROOT"
    return 1
  fi

  case $SHELL_NAME in
    "zsh")
      ZSHRC_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'tmp_zsh')
      echo "source $RC_FILE; source $OPENPILOT_ROOT/.venv/bin/activate" >> $ZSHRC_DIR/.zshrc
      ZDOTDIR=$ZSHRC_DIR zsh ;;
    *)
      bash --rcfile <(echo "source $RC_FILE; source $OPENPILOT_ROOT/.venv/bin/activate") ;;
  esac
}

function op_adb() {
  op_before_cmd
  op_run_command tools/scripts/adb_ssh.sh
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
  if [[ -f "/AGNOS" ]]; then
    # needed on AGNOS to not run out of memory
    op_run_command system/manager/build.py
  else
    # scons is fine on PC
    op_run_command scons $@
  fi
}

function op_juggle() {
  op_before_cmd
  op_run_command tools/plotjuggler/juggle.py $@
}

function op_lint() {
  op_before_cmd
  op_run_command scripts/lint/lint.sh $@
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

function op_clip() {
  op_before_cmd
  op_run_command tools/clip/run.py $@
}

function op_switch() {
  REMOTE="origin"
  if [ "$#" -gt 1 ]; then
    REMOTE="$1"
    shift
  fi

  if [ -z "$1" ]; then
    echo -e "${BOLD}${UNDERLINE}Usage:${NC} op switch [REMOTE] <BRANCH>"
    return 1
  fi
  BRANCH="$1"

  git fetch "$REMOTE" "$BRANCH"
  git checkout -f FETCH_HEAD
  git checkout -B "$BRANCH" --track "$REMOTE"/"$BRANCH"
  git reset --hard "${REMOTE}/${BRANCH}"
  git clean -df
  git submodule update --init --recursive
  git submodule foreach git reset --hard
  git submodule foreach git clean -df
}

function op_start() {
  if [[ -f "/AGNOS" ]]; then
    op_before_cmd
    op_run_command sudo systemctl restart comma $@
  fi
}

function op_stop() {
  if [[ -f "/AGNOS" ]]; then
    op_before_cmd
    op_run_command sudo systemctl stop comma $@
  fi
}

function op_default() {
  echo "An openpilot helper"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Description:${NC}"
  echo "  op is your entry point for all things related to openpilot development."
  echo "  op is only a wrapper for existing scripts, tools, and commands."
  echo "  op will always show you what it will run on your system."
  echo ""
  echo -e "${BOLD}${UNDERLINE}Usage:${NC} op [OPTIONS] <COMMAND>"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Commands [System]:${NC}"
  echo -e "  ${BOLD}auth${NC}         Authenticate yourself for API use"
  echo -e "  ${BOLD}check${NC}        Check the development environment (git, os, python) to start using openpilot"
  echo -e "  ${BOLD}venv${NC}         Activate the python virtual environment"
  echo -e "  ${BOLD}setup${NC}        Install openpilot dependencies"
  echo -e "  ${BOLD}build${NC}        Run the openpilot build system in the current working directory"
  echo -e "  ${BOLD}install${NC}      Install the 'op' tool system wide"
  echo -e "  ${BOLD}switch${NC}       Switch to a different git branch with a clean slate (nukes any changes)"
  echo -e "  ${BOLD}start${NC}        Starts (or restarts) openpilot"
  echo -e "  ${BOLD}stop${NC}         Stops openpilot"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Commands [Tooling]:${NC}"
  echo -e "  ${BOLD}juggle${NC}       Run PlotJuggler"
  echo -e "  ${BOLD}replay${NC}       Run Replay"
  echo -e "  ${BOLD}cabana${NC}       Run Cabana"
  echo -e "  ${BOLD}clip${NC}         Run clip (linux only)"
  echo -e "  ${BOLD}adb${NC}          Run adb shell"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Commands [Testing]:${NC}"
  echo -e "  ${BOLD}sim${NC}          Run openpilot in a simulator"
  echo -e "  ${BOLD}lint${NC}         Run the linter"
  echo -e "  ${BOLD}post-commit${NC}  Install the linter as a post-commit hook"
  echo -e "  ${BOLD}test${NC}         Run all unit tests from pytest"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Options:${NC}"
  echo -e "  ${BOLD}-d, --dir${NC}"
  echo "          Specify the openpilot directory you want to use"
  echo -e "  ${BOLD}--dry${NC}"
  echo "          Don't actually run anything, just print what would be run"
  echo -e "  ${BOLD}-n, --no-verify${NC}"
  echo "          Skip environment check before running commands"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Examples:${NC}"
  echo "  op setup"
  echo "          Run the setup script to install"
  echo "          openpilot's dependencies."
  echo ""
  echo "  op build -j4"
  echo "          Compile openpilot using 4 cores"
  echo ""
  echo "  op juggle --demo"
  echo "          Run PlotJuggler on the demo route"
}


function _op() {
  # parse Options
  case $1 in
    -d | --dir )       shift 1; OPENPILOT_ROOT="$1"; shift 1 ;;
    --dry )            shift 1; DRY="1" ;;
    -n | --no-verify ) shift 1; NO_VERIFY="1" ;;
    -l | --log )       shift 1; LOG_FILE="$1" ; shift 1 ;;
  esac

  # parse Commands
  case $1 in
    auth )          shift 1; op_auth "$@" ;;
    venv )          shift 1; op_venv "$@" ;;
    check )         shift 1; op_check "$@" ;;
    setup )         shift 1; op_setup "$@" ;;
    build )         shift 1; op_build "$@" ;;
    juggle )        shift 1; op_juggle "$@" ;;
    cabana )        shift 1; op_cabana "$@" ;;
    lint )          shift 1; op_lint "$@" ;;
    test )          shift 1; op_test "$@" ;;
    replay )        shift 1; op_replay "$@" ;;
    clip )          shift 1; op_clip "$@" ;;
    sim )           shift 1; op_sim "$@" ;;
    install )       shift 1; op_install "$@" ;;
    switch )        shift 1; op_switch "$@" ;;
    start )         shift 1; op_start "$@" ;;
    stop )          shift 1; op_stop "$@" ;;
    restart )       shift 1; op_restart "$@" ;;
    post-commit )   shift 1; op_install_post_commit "$@" ;;
    adb )           shift 1; op_adb "$@" ;;
    * ) op_default "$@" ;;
  esac
}

_op $@
