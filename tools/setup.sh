#!/usr/bin/env bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

if [ -z "$OPENPILOT_ROOT" ]; then
  # default to current directory for installation
  OPENPILOT_ROOT="$(pwd)/openpilot"
fi

function show_motd() {
cat << 'EOF'

                      .~ssos+.
                    +8888888888i,
                   {888888888888o.
                   h8888888888888k
                   t888888888s888k
                    `t88888d/ h88k
                       ```    h88l
                             ,88k`
                            .d8h`
                           +d8h
                        _+d8h`
                      ;y8h+`
                      |-`

              openpilot installer

EOF
}

function check_stdin() {
  echo "Checking for valid invocation..."
  if [ -t 0 ]; then
    echo -e " ↳ [${GREEN}✔${NC}] Installer successfully invoked\n"
    INTERACTIVE=1
  else
    echo -e " ↳ [${RED}✗${NC}] stdin not found! Running in non-interactive mode.\n"
  fi
}

function ask_dir() {
  if [[ -z $INTERACTIVE ]]; then
    return 0
  fi
  echo -n "Enter directory in which to install openpilot (default $OPENPILOT_ROOT): "
  read
  if [[ ! -z "$REPLY" ]]; then
    OPENPILOT_ROOT="$(realpath $REPLY/openpilot)"
  fi
}

function check_dir() {
  echo "Checking for installation directory..."
  if [ -d "$OPENPILOT_ROOT" ]; then
    echo -e " ↳ [${RED}✗${NC}] Installation destination $OPENPILOT_ROOT already exists !"
    if [[ -z $INTERACTIVE ]]; then
      return 1
    fi
    read -p "       Would you like to attempt installation anyway? [Y/n] " -n 1 -r
    echo -e "\n"
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      return 1
    else
      RETRY_INSTALLATION=1
      return 0
    fi
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
  st="$(date +%s)"
  echo "Cloning openpilot..."
  if $(git clone --filter=blob:none https://github.com/commaai/openpilot.git "$OPENPILOT_ROOT"); then
    if [[ -f $OPENPILOT_ROOT/launch_openpilot.sh ]]; then
      et="$(date +%s)"
      echo -e " ↳ [${GREEN}✔${NC}] Successfully cloned openpilot in $((et - st)) seconds.\n"
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

  echo -e "\n----------------------------------------------------------------------"
  echo -e "openpilot was successfully installed into ${BOLD}$OPENPILOT_ROOT${NC}"
  echo -e "Checkout the docs at https://docs.comma.ai"
  echo -e "Checkout how to contribute at https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md"
}

show_motd

check_stdin
ask_dir
check_dir
check_git
[ -z $RETRY_INSTALLATION ] && git_clone
install_with_op
