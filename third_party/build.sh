#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# Reproducible builds: pin timestamps to epoch
export SOURCE_DATE_EPOCH=0
export ZERO_AR_DATE=1

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

[ $failed -ne 0 ] && exit $failed

# Repack ar archives with deterministic headers (zero timestamps/uid/gid)
# Skip foreign-platform archives that ar can't read (e.g. Mach-O on Linux)
while IFS= read -r -d '' lib; do
  tmpdir=$(mktemp -d)
  lib=$(realpath "$lib")
  if (cd "$tmpdir" && ar x "$lib" 2>/dev/null); then
    (cd "$tmpdir" && ar Drcs repacked.a * && mv repacked.a "$lib")
  fi
  rm -rf "$tmpdir"
done < <(find "$DIR" -name '*.a' \
  \( -path '*/x86_64/*' -o -path '*/Darwin/*' -o -path '*/larch64/*' -o -path '*/aarch64/*' \) \
  -print0)

echo -e "\033[32mAll third_party builds succeeded.\033[0m"
