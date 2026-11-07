#!/usr/bin/env bash

BIG_FILES="$(git ls-tree -rl HEAD | awk '$4 > 95 * 1024 * 1024 {print $5}')"
if [ -n "$BIG_FILES" ]; then
  echo "Found Git blobs exceeding the 95 MiB release limit:"
  echo "$BIG_FILES"
  exit 1
fi
