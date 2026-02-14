#!/usr/bin/env bash
#
# Centralized platform/architecture detection for openpilot.
# Source this script to get OPENPILOT_ARCH set to one of:
#   larch64  - linux tici arm64
#   aarch64  - linux pc arm64
#   x86_64   - linux pc x64
#   Darwin   - macOS arm64
#

OPENPILOT_ARCH=$(uname -m)
if [[ "$OSTYPE" == "darwin"* ]]; then
  OPENPILOT_ARCH="Darwin"
elif [ -f /TICI ]; then
  OPENPILOT_ARCH="larch64"
fi


# unsupported platforms
if [[ "$OSTYPE" == "darwin"* && "$OPENPILOT_ARCH" == "x86_64" ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " ERROR: Intel-based Macs are not supported."
  echo ""
  echo " openpilot requires an Apple Silicon Mac (M1 or newer)."
  echo " See https://github.com/commaai/openpilot for supported"
  echo " hardware."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  exit 1
fi
