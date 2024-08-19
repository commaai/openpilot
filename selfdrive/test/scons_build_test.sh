#!/usr/bin/env bash

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
