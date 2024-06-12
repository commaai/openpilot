#!/usr/bin/env sh
set -e

scons -u

./canloader.py knee obj/body.bin.signed
