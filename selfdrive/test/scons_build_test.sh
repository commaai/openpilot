#!/bin/bash

# tests that our build system's dependencies are configured properly, 
# needs a machine with lots of cores
scons --clean
scons --no-cache --random -j$(nproc)