#!/bin/bash

set -Eeuo pipefail

# usage: hot.sh -- COMMAND [ARGS...]

command_exists() { command -v "$1" >/dev/null 2>&1; }

if ! command_exists inotifywait; then
  echo "error: inotifywait not found. attempting to install..." >&2
  if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command_exists brew; then
      echo "homebrew not found. install Homebrew and try again."
      exit 1
    fi
    brew install inotify-tools
  else
    sudo apt-get update -y || true
    sudo apt-get install -y inotify-tools
  fi
fi

resolve_repo_root() {
  if command_exists git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse --show-toplevel
  else
    pwd -P
  fi
}

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

ROOT_DIR="$(resolve_repo_root)"

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

while true; do
  start_child

  inotifywait -r --quiet --event modify,create,delete,move "$ROOT_DIR"

  if kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM -"$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
done
