#!/usr/bin/env bash

resolve_root_dir() {
  local root_dir
  root_dir="$(pwd -P)"

  # traverse up until we find launch_openpilot.sh
  while [[ "$root_dir" != "/" ]]; do
    if [[ -f "$root_dir/launch_openpilot.sh" ]]; then
      echo "$root_dir"
      return 0
    fi
    root_dir="$(readlink -f "$root_dir/..")"
  done

  # fallbacks
  for dir in "$HOME/openpilot" "/data/openpilot"; do
    if [[ -f "$dir/launch_openpilot.sh" ]]; then
      echo "$dir"
      return 0
    fi
  done

  pwd -P
}


