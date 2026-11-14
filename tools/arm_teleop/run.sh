#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
exec .venv/bin/python workstation.py "$@"
