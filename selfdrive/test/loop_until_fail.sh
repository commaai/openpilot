#!/usr/bin/env bash
set -e

# Loop something forever until it fails, for verifying new tests

count=$1

while true; do
  $@
done
