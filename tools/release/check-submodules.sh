#!/usr/bin/env bash

while read -r hash submodule _ref; do
  if [ "$submodule" = "tinygrad_repo" ]; then
    echo "Skipping $submodule"
    continue
  fi

  git -C "$submodule" fetch --depth 100 origin master
  if git -C "$submodule" branch -r --contains "$hash" | grep "origin/master"; then
    echo "$submodule ok"
  else
    echo "$submodule: $hash is not on master"
    exit 1
  fi
done <<< "$(git submodule status --recursive)"
