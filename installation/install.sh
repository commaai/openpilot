#!/bin/bash
set -e

# move files into place
adb push files/id_rsa_openpilot_ro /tmp/id_rsa_openpilot_ro
adb shell mv /tmp/id_rsa_openpilot_ro /data/data/com.termux/files/

# moving continue into place runs the continue script
adb push files/continue.sh /tmp/continue.sh
adb shell mv /tmp/continue.sh /data/data/com.termux/files/

