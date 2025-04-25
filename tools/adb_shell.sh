#!/usr/bin/env expect
spawn adb shell
expect "#"
send "cd data/openpilot\r"
send "export TERM=xterm-256color\r"
send "su comma\r"
send "clear\r"
interact
