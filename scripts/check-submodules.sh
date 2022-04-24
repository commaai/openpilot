#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

FILES=".pre-commit-config.yaml"

exit=0
while read hash submodule ref; do
  # ensure submodule linter files match
  for f in $FILES; do
    opf="$DIR/$f"
    subf="$DIR/$submodule/$f"
    if [ ! -f "$subf" ]; then
      echo "$submodule: $f missing"
      exit=1
    elif ! cmp --silent $opf $subf; then
      echo "$submodule: $f doesn't match openpilot"
      exit=1
    fi
  done

  # ensure submodule is on a commit from master
  git -C $submodule fetch --depth 100 origin master > /dev/null 2>&1
  git -C $submodule branch -r --contains $hash | grep "origin/master" > /dev/null 2>&1
  if [ "$?" -ne 0 ]; then
    echo "$submodule: $hash is not on master"
    exit=1
  fi
done <<< $(git submodule status --recursive)

exit $exit
