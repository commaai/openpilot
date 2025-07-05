#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../

set -x

COMMIT="$(grep tinygrad pyproject.toml | grep -oE '[0-9a-f]{40}')"

# package tinygrad for AGNOS
wget -O selfdrive/modeld/compile3.py \
  https://raw.githubusercontent.com/tinygrad/tinygrad/$COMMIT/examples/openpilot/compile3.py

uv sync
cp -r .venv/lib/python3.11/site-packages/tinygrad/ third_party/
