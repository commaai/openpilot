#!/usr/bin/env bash
set -e

# this is a little nicer than "adb shell" since
# "adb shell" doesn't do full terminal emulation
adb forward tcp:2222 tcp:22
ssh comma@localhost -p 2222
