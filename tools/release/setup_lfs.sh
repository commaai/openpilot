#!/usr/bin/env bash

if [ -n "$INCLUDE_BIG_MODEL" ]; then
  echo 'openpilot/selfdrive/modeld/models/big_*_tinygrad.pkl filter=lfs diff=lfs merge=lfs -text' > .gitattributes
fi

# Reset filters and hooks for the release checkout.
git config --local core.hooksPath "$(git rev-parse --git-common-dir)/hooks"
./lfs.py install
