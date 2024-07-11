#!/usr/bin/env bash

set -e

if [ ! -f launch_openpilot.sh ]; then
  if [ ! -d openpilot ]; then
    git clone --single-branch --recurse-submodules https://github.com/commaai/openpilot.git
  fi
  cd openpilot
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  tools/mac_setup.sh
else
  tools/ubuntu_setup.sh
fi

git lfs pull

source .venv/bin/activate

echo "Building openpilot"
scons -u -j$(nproc)

echo
echo "----   OPENPILOT BUILDING DONE   ----"
echo "To push changes to your fork, run the following commands:"
echo "git remote remove origin"
echo "git remote add origin git@github.com:<YOUR_USERNAME>/openpilot.git"
echo "git fetch"
echo "git commit -m \"first commit\""
echo "git push"
