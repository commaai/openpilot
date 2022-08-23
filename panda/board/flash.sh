#!/usr/bin/env sh
set -e

scons -u -j$(nproc)
printf %b 'from python import Panda\nfor serial in Panda.list(): Panda(serial).flash()' | PYTHONPATH=.. python3
