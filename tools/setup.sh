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

function sentry_send_event() {
  SENTRY_KEY=dd0cba62ba0ac07ff9f388f8f1e6a7f4
  SENTRY_URL=https://sentry.io/api/4507726145781760/store/

  EVENT=$1
  EVENT_TYPE=${2:-$EVENT}
  EVENT_LOG=${3:-"NA"}

  PLATFORM=$(uname -s)
  ARCH=$(uname -m)
  SYSTEM=$(uname -a)
  if [[ $PLATFORM == "Darwin" ]]; then
    OS="macos"
  elif [[ $PLATFORM == "Linux" ]]; then
    OS="linux"
  fi

  if [[ $ARCH == armv8* ]] || [[ $ARCH == arm64* ]] || [[ $ARCH == aarch64* ]]; then
    ARCH="aarch64"
  elif [[ $ARCH == "x86_64" ]] || [[ $ARCH == i686* ]]; then
    ARCH="x86"
  fi

  PYTHON_VERSION=$(echo $(python3 --version 2> /dev/null || echo "NA"))
  BRANCH=$(echo $(git -C $OPENPILOT_ROOT rev-parse --abbrev-ref HEAD 2> /dev/null || echo "NA"))
  COMMIT=$(echo $(git -C $OPENPILOT_ROOT rev-parse HEAD 2> /dev/null || echo "NA"))

  curl -s -o /dev/null -X POST -g --data "{ \"exception\": { \"values\": [{ \"type\": \"$EVENT\" }] }, \"tags\" : { \"event_type\" : \"$EVENT_TYPE\", \"event_log\" : \"$EVENT_LOG\", \"os\" : \"$OS\", \"arch\" : \"$ARCH\", \"python_version\" : \"$PYTHON_VERSION\" , \"git_branch\" : \"$BRANCH\", \"git_commit\" : \"$COMMIT\", \"system\" : \"$SYSTEM\" }  }" \
    -H 'Content-Type: application/json' \
    -H "X-Sentry-Auth: Sentry sentry_version=7, sentry_key=$SENTRY_KEY, sentry_client=op_setup/0.1" \
    $SENTRY_URL 2> /dev/null
}

function check_stdin() {
  if [ -t 0 ]; then
    INTERACTIVE=1
  else
    echo "Checking for valid invocation..."
    echo -e " ↳ [${RED}✗${NC}] stdin not found! Running in non-interactive mode."
    echo -e "       Run ${BOLD}'bash <(curl -fsSL openpilot.comma.ai)'${NC} to run in interactive mode.\n"
  fi
}

function ask_dir() {
  echo -n "Enter directory in which to install openpilot (default $OPENPILOT_ROOT): "

  if [[ -z $INTERACTIVE ]]; then
    echo -e "\nBecause your are running in non-interactive mode, the installation"
    echo -e "will default to $OPENPILOT_ROOT\n"
    return 0
  fi

  read
  if [[ ! -z "$REPLY" ]]; then
    mkdir -p $REPLY
    OPENPILOT_ROOT="$(realpath $REPLY)/openpilot"
  fi
}

function check_dir() {
  echo "Checking for installation directory..."
  if [ -d "$OPENPILOT_ROOT" ]; then
    echo -e " ↳ [${RED}✗${NC}] Installation destination $OPENPILOT_ROOT already exists!"

    # not a valid clone, can't continue
    if [[ ! -z "$(ls -A $OPENPILOT_ROOT)" && ! -f "$OPENPILOT_ROOT/launch_openpilot.sh" ]]; then
      echo -e "       $OPENPILOT_ROOT already contains files but does not seems"
      echo -e "       to be a valid openpilot git clone. Choose another location for"
      echo -e "       installing openpilot!\n"
      return 1
    fi

    # already a "valid" openpilot clone, skip cloning again
    if [[ ! -z "$(ls -A $OPENPILOT_ROOT)" ]]; then
      SKIP_GIT_CLONE=1
    fi

    # by default, don't try installing in already existing directory
    if [[ -z $INTERACTIVE ]]; then
      return 0
    fi

    read -p "       Would you like to attempt installation anyway? [Y/n] " -n 1 -r
    echo -e "\n"
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      return 1
    fi

    return 0
  fi

  echo -e " ↳ [${GREEN}✔${NC}] Successfully chosen $OPENPILOT_ROOT as installation directory\n"
}

function check_git() {
  echo "Checking for git..."
  if ! command -v "git" > /dev/null 2>&1; then
    echo -e " ↳ [${RED}✗${NC}] git not found on your system, can't continue!"
    sentry_send_event "SETUP_FAILURE" "ERROR_GIT_NOT_FOUND"
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
  sentry_send_event "SETUP_FAILURE" "ERROR_GIT_CLONE"
  return 1
}

function install_with_op() {
  cd $OPENPILOT_ROOT
  $OPENPILOT_ROOT/tools/op.sh install
  $OPENPILOT_ROOT/tools/op.sh post-commit

  LOG_FILE=$(mktemp)

  if ! $OPENPILOT_ROOT/tools/op.sh --log $LOG_FILE setup; then
    echo -e "\n[${RED}✗${NC}] failed to install openpilot!"

    ERROR_TYPE="$(cat "$LOG_FILE" | sed '1p;d')"
    ERROR_LOG="$(cat "$LOG_FILE" | sed '2p;d')"
    sentry_send_event "SETUP_FAILURE" "$ERROR_TYPE" "$ERROR_LOG" || true

    return 1
  else
    sentry_send_event "SETUP_SUCCESS" || true
  fi

  echo -e "\n----------------------------------------------------------------------"
  echo -e "[${GREEN}✔${NC}] openpilot was successfully installed into ${BOLD}$OPENPILOT_ROOT${NC}"
  echo -e "Checkout the docs at https://docs.comma.ai"
  echo -e "Checkout how to contribute at https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md"
}

show_motd
check_stdin
ask_dir
check_dir
check_git
[ -z $SKIP_GIT_CLONE ] && git_clone
install_with_op
