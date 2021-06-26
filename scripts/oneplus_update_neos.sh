#!/usr/bin/bash

if [ -z "$BASEDIR" ]; then
  BASEDIR="/data/openpilot"
fi

source "$BASEDIR/launch_env.sh"
cp -f "$BASEDIR/installer/updater/update.zip" "/sdcard/update.zip"
pm disable ai.comma.plus.offroad
killall _ui
"$BASEDIR/installer/updater/updater" "file://$BASEDIR/installer/updater/oneplus.json"
