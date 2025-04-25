#!/usr/bin/env bash
set -x

while read hash submodule ref; do
  if [ "$submodule" = "tinygrad_repo" ]; then
    echo "Skipping $submodule"
    continue
  fi

  if [ "$submodule" = "opendbc_repo" ]; then
    git -C $submodule fetch origin e932b98abd16554246012c79007657d8903805bd
    git -C $submodule checkout FETCH_HEAD
    cat opendbc_repo/opendbc/safety/tests/libsafety/safety_helpers.py | grep ignition
    ls -la opendbc
    echo "$submodule ok (custom Aubrey version)"
    continue
  fi

  if [ "$submodule" = "panda" ]; then
    git -C $submodule fetch origin c66e9a46349d152199a4a0997f82f0bea3c9b5a6
    git -C $submodule checkout FETCH_HEAD
    echo "$submodule ok (custom Aubrey version)"
    continue
  fi

  git -C $submodule fetch --depth 100 origin master
  git -C $submodule branch -r --contains $hash | grep "origin/master"
  if [ "$?" -eq 0 ]; then
    echo "$submodule ok"
  else
    echo "$submodule: $hash is not on master"
    exit 1
  fi
done <<< $(git submodule status --recursive)
