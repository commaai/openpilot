#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/../

IGNORED_FILES="uv\.lock|docs\/CARS.md"

function run() {
  echo -en "$1"

  for ((i=0; i<$((50 - ${#1})); i++)); do
    echo -n "."
  done

  shift 1;
  CMD="$@"

  ( set +e
  log="$((eval "$CMD" ) 2>&1)"

  if [[ $? -eq 0 ]]; then
    echo -e "[${GREEN}✔${NC}]"
  else
    echo -e "[${RED}✗${NC}]"
    echo "$log"
    return 1
  fi
  )
}

function run_tests() {
  echo "FILES:"
  echo $@
  echo "xxxxxxxxxxxx"

  ALL_FILES=$(echo "$@" | sed -E "s/$IGNORED_FILES//g")
  PYTHON_FILES=$(echo "$ALL_FILES" | grep --color=never '.py$' || true)

  echo $ALL_FILES
  echo $PYTHON_FILES

  if [[ -n "$PYTHON_FILES" ]]; then
    run "ruff" ruff check $PYTHON_FILES --quiet
  fi

  if [[ -n "$ALL_FILES" ]]; then
    run "Codespell" codespell $ALL_FILES
    run "Large files check" python3 -m pre_commit_hooks.check_added_large_files --enforce-all $ALL_FILES --maxkb=120
    run "Shebang check" python3 -m pre_commit_hooks.check_shebang_scripts_are_executable $ALL_FILES
  fi
}

case $1 in
  --files ) shift 1; FILES="$@" ;;
  --all )   shift 1; ALL="1" ;;
esac

if [[ -n $FILES ]]; then
  run_tests "$FILES"
else
  run_tests "$(git diff --name-only --cached --diff-filter=AM $(git merge-base HEAD master))"
fi
