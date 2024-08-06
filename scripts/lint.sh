#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/../

IGNORED_FILES="uv\.lock|docs\/CARS.md"

PYTHON_FILES=$(git diff --name-only origin/master --diff-filter=AM | grep --color=never '.py$' || true)
ALL_FILES=$(git diff --name-only origin/master --diff-filter=AM | sed -E "s/$IGNORED_FILES//g")

if [[ -n "$ALL_FILES" ]]; then
  codespell $ALL_FILES
  python3 -m pre_commit_hooks.check_added_large_files --enforce-all $ALL_FILES --maxkb=120
  python3 -m pre_commit_hooks.check_shebang_scripts_are_executable $ALL_FILES
fi

if [[ -n "$PYTHON_FILES" ]]; then
  ruff check $PYTHON_FILES
fi
