#!/bin/bash
set -e

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# TODO: why doesn't uv do this?
export PYTHONPATH=$BASEDIR

# *** dependencies install ***
if ! command -v uv &>/dev/null; then
  echo "'uv' is not installed. Installing 'uv'..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export UV_PROJECT_ENVIRONMENT="$BASEDIR/.venv"
uv sync --all-extras
source "$PYTHONPATH/.venv/bin/activate"

$BASEDIR/opendbc/safety/tests/misra/install.sh
