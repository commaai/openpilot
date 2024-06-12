#!/usr/bin/env sh
set -e

scons -u

./canloader.py base obj/body.bin.signed
