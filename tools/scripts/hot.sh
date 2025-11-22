#!/usr/bin/env bash

# usage: hot.sh -- COMMAND [ARGS...]

set -Eeuo pipefail

TOOLS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source "$TOOLS_DIR/lib/common.sh"

command_exists() { command -v "$1" >/dev/null 2>&1; }

if ! command_exists fswatch; then
  echo "error: fswatch not found. did you run setup?" >&2
  exit 1
fi

CMD=( )
if [[ $# -eq 0 || "$1" != "--" ]]; then
  echo "usage: $(basename "$0") -- COMMAND [ARGS...]" >&2
  exit 2
fi
shift # drop --
if [[ $# -eq 0 ]]; then
  echo "error: missing command to run" >&2
  exit 2
fi
CMD=("$@")

ROOT_DIR="$(resolve_root_dir)"

echo "watching: $ROOT_DIR" >&2
echo "command: ${CMD[*]}" >&2

child_pid=""
cleanup() {
  if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM -"$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

start_child() {
  # start in its own process group so we can kill descendants
  setsid "${CMD[@]}" &
  child_pid=$!
}

exclude_flags=""
if [[ -f "$ROOT_DIR/.gitignore" ]]; then
  exclude_flags=$(cd "$ROOT_DIR" && grep -v '^[[:space:]]*#' .gitignore | grep -v '^[[:space:]]*$' | grep -v '^!' | sed 's|/$||' | sed 's|^|-e |')
fi

while true; do
  start_child

  echo "watching: $ROOT_DIR" >&2
  fswatch -1 --latency=1 --event Updated --event Removed --event Renamed -x -Lr openpilot/ | while read event; do echo "$event"; done

  if kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM -"$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
done
