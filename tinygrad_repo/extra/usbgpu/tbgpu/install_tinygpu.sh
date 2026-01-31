#!/bin/bash
set -e

APP_PATH="/Applications/TinyGPU.app"
DEXT_ID="org.tinygrad.tinygpu.edriver"

# Install app if not present. TODO: url
if [[ ! -d "$APP_PATH" ]]; then
  echo "TinyGPU.app not found in /Applications"
  exit 1
fi

# Ask user to install
read -n1 -p "Install TinyGPU driver extension now? [y/N] " answer
echo

if [[ "$answer" =~ ^[Yy]$ ]]; then
  "$APP_PATH/Contents/MacOS/TinyGPU" install
else
  echo "Skipped."
  exit 0
fi
