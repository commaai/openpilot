#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

pids=()
names=()
logs=()

for script in "$DIR"/*/build.sh; do
  [ -f "$script" ] || continue
  name=$(basename "$(dirname "$script")")
  log=$(mktemp)
  names+=("$name")
  logs+=("$log")
  (cd "$(dirname "$script")" && bash "$(basename "$script")") >"$log" 2>&1 &
  pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
  echo "--- ${names[$i]} ---"
  if wait "${pids[$i]}"; then
    echo "OK"
  else
    echo "FAILED (exit $?)"
    failed=1
  fi
  cat "${logs[$i]}"
  rm -f "${logs[$i]}"
  echo
done

exit $failed
