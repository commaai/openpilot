#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

for script in "$DIR"/*/build.sh; do
  [ -f "$script" ] || continue
  echo "--- Building $(basename "$(dirname "$script")") ---"
  "$script"
done
