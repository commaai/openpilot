#!/bin/bash

set -e

RED="\033[0;31m"
GREEN="\033[0;32m"
CLEAR="\033[0m"

BRANCHES="devel dashcam3 release3"
for b in $BRANCHES; do
  if git diff --quiet origin/$b origin/$b-staging && [ "$(git rev-parse origin/$b)" = "$(git rev-parse origin/$b-staging)" ]; then
    printf "%-10s $GREEN ok $CLEAR\n" "$b"
  else
    printf "%-10s $RED mismatch $CLEAR\n" "$b"
  fi
done
