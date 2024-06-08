#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

export SUDO=""

# Use sudo if not root
if [[ ! $(id -u) -eq 0 ]]; then
  if [[ -z $(which sudo) ]]; then
    echo "Please install sudo or run as root"
    exit 1
  fi
  SUDO="sudo"
fi

clear
cat <<"EOF"
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

EOF


echo " --   WELCOME TO THE OPENPILOT SETUP   --"
echo
echo "-- sudo is required for apt installation --"

# NOTE: this is used in a docker build, so do not run any scripts here.

$DIR/install_ubuntu_dependencies.sh
$DIR/install_python_dependencies.sh

clear
echo
echo "----   OPENPILOT SETUP DONE   ----"
echo "Open a new shell or configure your active shell env by running:"
echo "source ~/.bashrc"
echo
echo "To activate your virtual env using poetry, run either:"
echo
echo "\`poetry shell\` or \`.venv/bin/activate\`"
