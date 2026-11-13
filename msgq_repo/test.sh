#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source ./setup.sh

# *** build ***
scons -j8

# *** lint + test ***
lefthook run test

# *** test installed package outside the checkout ***
(
  TEST_DIR=$(mktemp -d)
  trap 'rm -rf "$TEST_DIR"' EXIT
  uv venv --python "$DIR/.venv/bin/python" "$TEST_DIR/.venv"
  uv pip install --python "$TEST_DIR/.venv/bin/python" "$DIR"
  cd "$("$TEST_DIR/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_path('purelib'))")"
  "$TEST_DIR/.venv/bin/python" -m unittest discover -t . -s msgq
)

# *** all done ***
GREEN='\033[0;32m'
NC='\033[0m'
printf "\n${GREEN}All good!${NC} Finished build, lint, and test in ${SECONDS}s\n"
