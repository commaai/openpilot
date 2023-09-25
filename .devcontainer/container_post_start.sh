#!/usr/bin/env bash

# setup safe directories for submodules
SUBMODULE_DIRS=$(git config --file .gitmodules --get-regexp path | awk '{ print $2 }')
for DIR in $SUBMODULE_DIRS; do 
  git config --global --add safe.directory "$PWD/$DIR"
done
