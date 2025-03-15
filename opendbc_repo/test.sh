#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source ./setup.sh

# *** build ***
scons -j8

# *** lint ***
# TODO: pre-commit is slow; replace it with openpilot's "op lint"
#pre-commit run --all-files
ruff check .

# *** test ***
pytest -n8 --ignore opendbc/safety

# *** all done ***
GREEN='\033[0;32m'
NC='\033[0m'
printf "\n${GREEN}All good!${NC} Finished build, lint, and test in ${SECONDS}s\n"
