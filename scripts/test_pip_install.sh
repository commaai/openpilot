#!/usr/bin/env bash
set -euxo pipefail

PACKAGE="${1:-git+https://github.com/commaai/openpilot.git}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cd $TMPDIR

uv venv
source .venv/bin/activate
uv pip install "$PACKAGE"
python3 - <<'PY'
from openpilot.tools.lib.logreader import LogReader

assert LogReader.__name__ == "LogReader"
print("ok: imported LogReader")
PY
