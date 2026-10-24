#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source ./setup.sh

# *** uv lockfile check ***
uv lock --check

# *** lint + test ***
lefthook run test

# *** all done ***
GREEN='\033[0;32m'
NC='\033[0m'
printf "\n${GREEN}All good!${NC} Finished lint and test in ${SECONDS}s\n"
