#!/usr/bin/env bash

resolve_root_dir() {
  local root_dir
  root_dir="$(pwd -P)"

  # First try traversing up the directory tree
  while [[ "$root_dir" != "/" ]]; do
    if [[ -f "$root_dir/launch_openpilot.sh" ]]; then
      echo "$root_dir"
      return 0
    fi
    root_dir="$(readlink -f "$root_dir/..")"
  done

  # Fallback to hardcoded directories if not found
  for dir in "$HOME/openpilot" "/data/openpilot"; do
    if [[ -f "$dir/launch_openpilot.sh" ]]; then
      echo "$dir"
      return 0
    fi
  done

  pwd -P
}
