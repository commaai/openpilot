#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
PYTHON_VERSION="$(cat "$REPO_ROOT/.python-version")"
PACKAGE="${1:-$REPO_ROOT}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cd "$TMPDIR"

uv venv --python "$PYTHON_VERSION"
source .venv/bin/activate
uv pip install "$PACKAGE"
python3 - <<'PY'
from openpilot.tools.lib.logreader import LogReader

assert LogReader.__name__ == "LogReader"
print("ok: imported LogReader")
PY
