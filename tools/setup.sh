#!/usr/bin/env bash

set -e

if [ ! -f launch_openpilot.sh ]; then
  if [ ! -d openpilot ]; then
    git clone --recurse-submodules https://github.com/commaai/openpilot.git
    git lfs pull
  fi
  cd openpilot
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  tools/mac_setup.sh
else
  tools/ubuntu_setup.sh
fi

source .venv/bin/activate

echo "Building openpilot"
scons -u -j$(nproc)
