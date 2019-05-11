#!/usr/bin/bash

if [ -z "$PASSIVE" ]; then
  export PASSIVE="1"
fi

function launch {
  # apply update only if no_ota_updates does not exist in /data directory
  file="/data/no_ota_updates"
  if ! [ -f "$file" ]; then
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse @{u})" ]; then
      git reset --hard @{u} &&
      git clean -xdf &&
      exec "${BASH_SOURCE[0]}"
    fi
  fi

  # no cpu rationing for now
  echo 0-3 > /dev/cpuset/background/cpus
  echo 0-3 > /dev/cpuset/system-background/cpus
  echo 0-3 > /dev/cpuset/foreground/boost/cpus
  echo 0-3 > /dev/cpuset/foreground/cpus
  echo 0-3 > /dev/cpuset/android/cpus

  # handle pythonpath
  ln -s /data/openpilot /data/pythonpath
  export PYTHONPATH="$PWD"

  # start manager
  cd selfdrive
  ./manager.py
  #./manager.py > log.txt

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
