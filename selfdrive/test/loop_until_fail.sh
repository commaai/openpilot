#!/usr/bin/env bash
set -e

# Loop something forever until it fails, for verifying new tests

while true; do
  $@
done
