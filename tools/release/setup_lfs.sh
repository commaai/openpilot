#!/usr/bin/env bash

if [ -n "$INCLUDE_BIG_MODEL" ]; then
  echo 'openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl filter=lfs diff=lfs merge=lfs -text' > .gitattributes
fi

# Reset filters and hooks since releases exclude .venv.
git config --local core.hooksPath "$(git rev-parse --git-common-dir)/hooks"
git lfs install --local --force
