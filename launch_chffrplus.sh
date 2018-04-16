#!/usr/bin/bash

if [ -z "$PASSIVE" ]; then
  export PASSIVE="1"
fi

function launch {
  DO_UPDATE=$(cat /data/params/d/ShouldDoUpdate)
  # apply update
  if [ "$DO_UPDATE" == "1" ] && [ "$(git rev-parse HEAD)" != "$(git rev-parse @{u})" ]; then
     git reset --hard @{u} &&
     git clean -xdf &&
     exec "${BASH_SOURCE[0]}"
     echo -n 0 > /data/params/d/ShouldDoUpdate
     echo -n 0 > /data/params/d/IsUpdateAvailable
  fi

  # no cpu rationing for now
  echo 0-3 > /dev/cpuset/background/cpus
  echo 0-3 > /dev/cpuset/system-background/cpus
  echo 0-3 > /dev/cpuset/foreground/boost/cpus
  echo 0-3 > /dev/cpuset/foreground/cpus
  echo 0-3 > /dev/cpuset/android/cpus

  # check if NEOS update is required
  while [ "$(cat /VERSION)" -lt 4 ] && [ ! -e /data/media/0/noupdate ]; do
    # wait for network
    (cd selfdrive/ui/spinner && exec ./spinner 'waiting for network...') & spin_pid=$!
    until ping -W 1 -c 1 8.8.8.8; do sleep 1; done
    kill $spin_pid

    # update NEOS
    curl -o /tmp/updater https://neos.comma.ai/updater && chmod +x /tmp/updater && /tmp/updater
    sleep 10
  done

  export PYTHONPATH="$PWD"

  # start manager
  cd selfdrive
  ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
