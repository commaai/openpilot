#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$PASSIVE" ]; then
  export PASSIVE="1"
fi

function launch {
  # Wifi scan
  wpa_cli IFNAME=wlan0 SCAN

  # no cpu rationing for now
  echo 0-3 > /dev/cpuset/background/cpus
  echo 0-3 > /dev/cpuset/system-background/cpus
  echo 0-3 > /dev/cpuset/foreground/boost/cpus
  echo 0-3 > /dev/cpuset/foreground/cpus
  echo 0-3 > /dev/cpuset/android/cpus

  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

  # Remove old NEOS update file
  if [ -d /data/neoupdate ]; then
    rm -rf /data/neoupdate
  fi

  # Check for NEOS update
  if [ $(< /VERSION) != "13" ]; then
    if [ -f "$DIR/scripts/continue.sh" ]; then
      cp "$DIR/scripts/continue.sh" "/data/data/com.termux/files/continue.sh"
    fi

    git clean -xdf
    "$DIR/installer/updater/updater" "file://$DIR/installer/updater/update.json"
  fi


  # handle pythonpath
  ln -sfn $(pwd) /data/pythonpath
  export PYTHONPATH="$PWD"

  # start manager
  cd selfdrive
  ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
