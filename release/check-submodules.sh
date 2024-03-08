#!/bin/bash

while read hash submodule ref; do
  if [ "$submodule" == "tinygrad_repo" ]; then
    # Skip checks for non-comma submodules
    continue
  fi
  git -C $submodule fetch --depth 2000 origin master
  git -C $submodule branch -r --contains $hash | grep "origin/master"
  if [ "$?" -eq 0 ]; then
    echo "$submodule ok"
  else
    echo "$submodule: $hash is not on master"
    exit 1
  fi
done <<< $(git submodule status --recursive)
