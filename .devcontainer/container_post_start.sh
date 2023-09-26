#!/usr/bin/env bash

# setup safe directories for submodules
SUBMODULE_DIRS=$(git config --file .gitmodules --get-regexp path | awk '{ print $2 }')
for DIR in $SUBMODULE_DIRS; do 
  git config --global --add safe.directory "$PWD/$DIR"
done

TARGET_USER=batman

# These lines are temporary, to remain backwards compatible with old devcontainers
# that were running as root and therefore had their caches written as root
sudo chown -R $TARGET_USER: /tmp/scons_cache
sudo chown -R $TARGET_USER: /tmp/comma_download_cache