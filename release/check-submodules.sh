#!/bin/bash

while read hash submodule ref; do
  git -C $submodule fetch --depth 100 origin master
  git -C $submodule branch -r --contains $hash | grep "origin/master"
  if [ "$?" -eq 0 ]; then
    echo "$submodule ok"
  else
    echo "$submodule: $hash is not on master"
    exit 1
  fi
done <<< $(git submodule status --recursive)
