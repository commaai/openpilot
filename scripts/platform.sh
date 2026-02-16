#!/usr/bin/env bash
#
# Centralized platform and architecture detection for openpilot.
# Source this script to get OPENPILOT_ARCH set to one of:
#   larch64  - linux tici arm64
#   aarch64  - linux pc arm64
#   x86_64   - linux pc x64
#   Darwin   - macOS arm64
#

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

OPENPILOT_ARCH=$(uname -m)

# ── check OS and normalize arch ──────────────────────────────
if [ -f /TICI ]; then
  # TICI runs AGNOS — no OS validation needed
  OPENPILOT_ARCH="larch64"

elif [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ "$OPENPILOT_ARCH" == "x86_64" ]]; then
    echo -e " ↳ [${RED}✗${NC}] Intel-based Macs are not supported!"
    echo "       openpilot requires an Apple Silicon Mac (M1 or newer)."
    exit 1
  fi
  echo -e " ↳ [${GREEN}✔${NC}] macOS detected."
  OPENPILOT_ARCH="Darwin"

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  if [ -f "/etc/os-release" ]; then
    source /etc/os-release
    case "$VERSION_CODENAME" in
      "jammy" | "kinetic" | "noble" | "focal")
        echo -e " ↳ [${GREEN}✔${NC}] Ubuntu $VERSION_CODENAME detected."
        ;;
      *)
        echo -e " ↳ [${RED}✗${NC}] Incompatible Ubuntu version $VERSION_CODENAME detected!"
        exit 1
        ;;
    esac
  else
    echo -e " ↳ [${RED}✗${NC}] No /etc/os-release on your system. Make sure you're running on Ubuntu, or similar!"
    exit 1
  fi

else
  echo -e " ↳ [${RED}✗${NC}] OS type $OSTYPE not supported!"
  exit 1
fi
