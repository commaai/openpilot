#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR/..

# ensure files are within GitHub's limit
BIG_FILES="$(find . -type f -not -path './.git/*' -size +95M)"
if [ ! -z "$BIG_FILES" ]; then
  printf '\n\n\n'
  echo "Found files exceeding GitHub's 100MB limit:"
  echo "$BIG_FILES"
  exit 1
fi
