#!/usr/bin/env bash

FAIL=0

if grep -n '\(#\|//\)\([[:space:]]*\)NOMERGE' $@; then
  echo -e "NOMERGE comments found! Remove them before merging\n"
  FAIL=1
fi

exit $FAIL
