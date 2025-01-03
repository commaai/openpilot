#!/usr/bin/env bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
UNDERLINE='\033[4m'
BOLD='\033[1m'
NC='\033[0m'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR/../../"
cd $ROOT

FAILED=0

IGNORED_FILES="uv\.lock|docs\/CARS.md|LICENSE\.md|.*\.zst"
IGNORED_DIRS="^third_party.*|^msgq.*|^msgq_repo.*|^opendbc.*|^opendbc_repo.*|^cereal.*|^panda.*|^rednose.*|^rednose_repo.*|^tinygrad.*|^tinygrad_repo.*|^teleoprtc.*|^teleoprtc_repo.*"

function run() {
  shopt -s extglob
  case $1 in
    $SKIP | $RUN ) return 0 ;;
  esac

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
  ALL_FILES=$1
  PYTHON_FILES=$2

  run "ruff" ruff check $ROOT --quiet
  run "check_added_large_files" python3 -m pre_commit_hooks.check_added_large_files --enforce-all $ALL_FILES --maxkb=120
  run "check_shebang_scripts_are_executable" python3 -m pre_commit_hooks.check_shebang_scripts_are_executable $ALL_FILES
  run "check_shebang_format" $DIR/check_shebang_format.sh $ALL_FILES
  run "check_nomerge_comments" $DIR/check_nomerge_comments.sh $ALL_FILES
  run "check_raylib_includes" $DIR/check_raylib_includes.sh $ALL_FILES

  if [[ -z "$FAST" ]]; then
    run "mypy" mypy $PYTHON_FILES
    run "codespell" codespell $ALL_FILES --ignore-words=$ROOT/.codespellignore
  fi

  return $FAILED
}

function help() {
  echo "A fast linter"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Usage:${NC} op lint [TESTS] [OPTIONS]"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Tests:${NC}"
  echo -e "  ${BOLD}ruff${NC}"
  echo -e "  ${BOLD}mypy${NC}"
  echo -e "  ${BOLD}codespell${NC}"
  echo -e "  ${BOLD}check_added_large_files${NC}"
  echo -e "  ${BOLD}check_shebang_scripts_are_executable${NC}"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Options:${NC}"
  echo -e "  ${BOLD}-f, --fast${NC}"
  echo "          Skip slow tests"
  echo -e "  ${BOLD}-s, --skip${NC}"
  echo "          Specify tests to skip separated by spaces"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Examples:${NC}"
  echo "  op lint mypy ruff"
  echo "          Only run the mypy and ruff tests"
  echo ""
  echo "  op lint --skip mypy ruff"
  echo "          Skip the mypy and ruff tests"
  echo ""
  echo "  op lint"
  echo "          Run all the tests"
}

SKIP=""
RUN=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -f | --fast ) shift 1; FAST="1" ;;
    -s | --skip ) shift 1; SKIP=" " ;;
    -h | --help | -help | --h ) help; exit 0 ;;
    * ) if [[ -n $SKIP ]]; then SKIP+="$1 "; else RUN+="$1 "; fi; shift 1 ;;
  esac
done

RUN=$([ -z "$RUN" ] && echo "" || echo "!($(echo $RUN | sed 's/ /|/g'))")
SKIP="@($(echo $SKIP | sed 's/ /|/g'))"

GIT_FILES="$(git ls-files | sed -E "s/$IGNORED_FILES|$IGNORED_DIRS//g")"
ALL_FILES=""
for f in $GIT_FILES; do
  if [[ -f $f ]]; then
    ALL_FILES+="$f"$'\n'
  fi
done
PYTHON_FILES=$(echo "$ALL_FILES" | grep --color=never '.py$' || true)

run_tests "$ALL_FILES" "$PYTHON_FILES"
