#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/../

FAILED=0
REF_BRANCH="master"

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
  ALL_FILES=$(echo "$@" | sed -E "s/$IGNORED_FILES|$IGNORED_DIRS//g")
  PYTHON_FILES=$(echo "$ALL_FILES" | grep --color=never '.py$' || true)

  echo $ALL_FILES
  echo $PYTHON_FILES

  if [[ -n "$PYTHON_FILES" ]]; then
    run "ruff" ruff check $PYTHON_FILES --quiet
    run "mypy" mypy $PYTHON_FILES
    run "Import check" lint-imports
  fi

  if [[ -n "$ALL_FILES" ]]; then
    run "Codespell" codespell $ALL_FILES
    run "Large files check" python3 -m pre_commit_hooks.check_added_large_files --enforce-all $ALL_FILES --maxkb=120
    run "Shebang check" python3 -m pre_commit_hooks.check_shebang_scripts_are_executable $ALL_FILES
  fi

  return $FAILED
}

case $1 in
  -b | --branch ) shift 1; REF_BRANCH="$1"; shift 1 ;;
esac

# run against the given existing files
if [[ -n $@ ]]; then
  FILES=""
  for f in $@; do
    if [[ -f "$f" ]]; then
      FILES+="$f"$'\n'
    fi
  done
  run_tests "$FILES"

# run against the the diff between HEAD and REF_BRANCH (default to master)
else
  ANCESTOR=$(git merge-base HEAD $REF_BRANCH || echo "")
  if [[ -z $ANCESTOR ]]; then
    echo -e "[${RED}✗${NC}] No common commit found between HEAD and $REF_BRANCH"
    exit 1
  fi
  STAGED_FILES="git diff --name-only --staged --diff-filter=AM $ANCESTOR"
  UNSTAGED_FILES="git diff --name-only --diff-filter=AM $ANCESTOR"
  FILES=$({ $STAGED_FILES ; $UNSTAGED_FILES ; } | sort | uniq )
  echo $FILES
  run_tests "$FILES"
fi
