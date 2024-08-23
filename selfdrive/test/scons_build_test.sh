#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(dirname "$0")
BASEDIR=$(realpath "$SCRIPT_DIR/../../")
cd $BASEDIR

# tests that our build system's dependencies are configured properly,
# needs a machine with lots of cores

# helpful commands:
# scons -Q --tree=derived

cd $BASEDIR/opendbc_repo/
scons --clean
scons --no-cache --random -j$(nproc)
if ! scons -q; then
  echo "FAILED: all build products not up to date after first pass."
  exit 1
fi
