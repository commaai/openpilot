#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
BASEDIR=$(realpath "$SCRIPT_DIR/../../")
cd $BASEDIR

# tests that our build system's dependencies are configured properly, 
# needs a machine with lots of cores
scons --clean
scons --no-cache --random -j$(nproc)