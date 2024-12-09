#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# ensure we're up to date
uv sync --all-extras
source .venv/bin/activate

uv run scons -j8

uv run pre-commit run --all-files
uv run pytest -n8

GREEN='\033[0;32m'
NC='\033[0m'
printf "\n${GREEN}All good!${NC} Finished build, lint, and test in ${SECONDS}s\n"
