#!/usr/bin/env bash

function fail {
  echo $1 >&2
  exit 1
}

function retry {
  local n=1
  local max=3 # 3 retries before failure
  local delay=5 # delay between retries, 5 seconds
  while true; do
    echo "Running command '$@' with retry, attempt $n/$max"
    "$@" && break || {
      if [[ $n -lt $max ]]; then
        ((n++))
        sleep $delay;
      else
        fail "The command has failed after $n attempts."
      fi
    }
  done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    retry "$@"
fi
