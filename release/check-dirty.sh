#!/usr/bin/env bash

# exit script if any command fails
set -e

# navigate to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# check if there are any uncommitted changes
if git status --porcelain; then
  echo "Dirty working tree after build:"
  git status --porcelain
  exit 1
fi
