#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/../

FAILED=0

IGNORED_FILES="uv\.lock|docs\/CARS.md"
IGNORED_DIRS="^third_party.*|^msgq.*|^msgq_repo.*|^opendbc.*|^opendbc_repo.*|^panda.*|^rednose.*|^rednose_repo.*|^tinygrad.*|^tinygrad_repo.*|^teleoprtc.*|^teleoprtc_repo.*"

function run() {
  echo -en "$1"

  for ((i=0; i<$((50 - ${#1})); i++)); do
    echo -n "."
  done

  shift 1;
  CMD="$@"

  set +e
  log="$((eval "$CMD" ) 2>&1)"

  if [[ $? -eq 0 ]]; then
    echo -e "[${GREEN}✔${NC}]"
  else
    echo -e "[${RED}✗${NC}]"
    echo "$log"
    FAILED=1
  fi
  set -e
}

function run_tests() {
  run "ruff" ruff check . --quiet
  run "Import check" lint-imports
  run "Large files check" python3 -m pre_commit_hooks.check_added_large_files --enforce-all $@ --maxkb=120
  run "Shebang check" python3 -m pre_commit_hooks.check_shebang_scripts_are_executable $@

  if [[ -z "$FAST" ]]; then
    run "mypy" mypy -v .
    run "Codespell" codespell
  fi

  return $FAILED
}

case $1 in
  -f | --fast ) shift 1; FAST="1" ;;
esac

GIT_FILES="$(git ls-files | sed -E "s/$IGNORED_FILES|$IGNORED_DIRS//g")"
FILES=""
for f in $GIT_FILES; do
  if [[ -f $f ]]; then
    FILES+="$f "
  fi
done

run_tests "$FILES"
