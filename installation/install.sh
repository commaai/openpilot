#!/bin/bash
set -e

# moving continue into place runs the continue script
adb push files/continue.sh /tmp/continue.sh
adb shell mv /tmp/continue.sh /data/data/com.termux/files/

