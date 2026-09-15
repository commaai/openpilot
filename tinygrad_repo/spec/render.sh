#!/bin/bash
set -e

if ! command -v tectonic &>/dev/null; then
  echo "tectonic not found, installing..."
  sudo pacman -S --noconfirm tectonic
fi

tectonic tinyspec.tex
echo "done: tinyspec.pdf"
