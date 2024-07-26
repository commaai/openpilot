#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

if [ -z "$OPENPILOT_ROOT" ]; then
  # default to current directory for installation
  OPENPILOT_ROOT="$(pwd)/openpilot"
fi

function check_dir() {
  echo "Checking for installation directory..."
  if [ -d "$OPENPILOT_ROOT" ]; then
    echo -e " ↳ [${RED}✗${NC}] can't install openpilot in $OPENPILOT_ROOT !"
    return 1
  fi

  echo -e " ↳ [${GREEN}✔${NC}] Successfully chosen $OPENPILOT_ROOT as installation directory\n"
}

function check_git() {
  echo "Checking for git..."
  if ! command -v "git" > /dev/null 2>&1; then
    echo -e " ↳ [${RED}✗${NC}] git not found on your system, can't continue!"
    return 1
  else
    echo -e " ↳ [${GREEN}✔${NC}] git found.\n"
  fi
}

function git_clone() {
  echo "Cloning openpilot..."
  if $(git clone --filter=blob:none https://github.com/commaai/openpilot.git "$OPENPILOT_ROOT"); then
    if [[ -f $OPENPILOT_ROOT/launch_openpilot.sh ]]; then
      echo -e " ↳ [${GREEN}✔${NC}] Successfully cloned openpilot.\n"
      return 0
    fi
  fi

  echo -e " ↳ [${RED}✗${NC}] failed to clone openpilot!"
  return 1
}

function install_with_op() {
  cd $OPENPILOT_ROOT
  $OPENPILOT_ROOT/tools/op.sh install
  $OPENPILOT_ROOT/tools/op.sh setup

  # make op usable right now
  alias op="source $OPENPILOT_ROOT/tools/op.sh \"\$@\""
}

check_dir && check_git && git_clone && install_with_op

unset OPENPILOT_ROOT
unset RED
unset GREEN
unset NC
