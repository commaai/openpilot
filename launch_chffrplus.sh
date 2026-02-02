#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$DIR/launch_env.sh"

function agnos_init {
  # TODO: move this to agnos
  sudo rm -f /data/etc/NetworkManager/system-connections/*.nmmeta

  # set success flag for current boot slot
  sudo abctl --set_success

  # TODO: do this without udev in AGNOS
  # udev does this, but sometimes we startup faster
  sudo chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
  sudo chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0

  # Check if AGNOS update is required
  if [ $(< /VERSION) != "$AGNOS_VERSION" ]; then
    AGNOS_PY="$DIR/system/hardware/tici/agnos.py"
    MANIFEST="$DIR/system/hardware/tici/agnos.json"
    if $AGNOS_PY --verify $MANIFEST; then
      sudo reboot
    fi
    $DIR/system/hardware/tici/updater $AGNOS_PY $MANIFEST
  fi
}

function init_ble {
  if [ ! -e /dev/ttyHS1 ]; then
    return
  fi

  # Add system packages (gi/GLib/dbus) to venv so ble.py can import them
  PTH_FILE="/usr/local/venv/lib/python3.12/site-packages/system-packages.pth"
  if [ ! -f "$PTH_FILE" ]; then
    echo "Adding system packages to venv..."
    sudo mount -o remount,rw / 2>/dev/null || true
    echo '/usr/lib/python3/dist-packages' | sudo tee "$PTH_FILE" > /dev/null
  fi

  # If btattach is already running and adapter is powered, skip re-init
  if pgrep -f "btattach.*ttyHS1" >/dev/null 2>&1 && hciconfig hci0 >/dev/null 2>&1; then
    echo "Bluetooth already running"
    return
  fi

  echo "Initializing Bluetooth..."
  sudo pkill -f btattach 2>/dev/null || true
  sudo hciconfig hci0 down 2>/dev/null || true
  sleep 1

  # Run btattach in its own session so it survives manager/launch restarts
  sudo setsid btattach -B /dev/ttyHS1 -S 115200 </dev/null &>/dev/null &

  # wait for adapter
  for i in $(seq 1 10); do
    sleep 1
    if hciconfig hci0 >/dev/null 2>&1; then
      echo "Bluetooth adapter found"
      sudo hciconfig hci0 down

      # set static address derived from DongleId
      DONGLE_ID=$(cat /data/params/d/DongleId 2>/dev/null)
      if [ -n "$DONGLE_ID" ]; then
        DID=$(echo "$DONGLE_ID" | tr '[:upper:]' '[:lower:]')
        MAC="C0:${DID:0:2}:${DID:2:2}:${DID:4:2}:${DID:6:2}:${DID:8:2}"
        sudo btmgmt --index 0 static-addr "$MAC" 2>/dev/null || true
        sudo btmgmt --index 0 privacy on 2>/dev/null || true
      fi

      sudo hciconfig hci0 up
      echo "Bluetooth initialized"
      return
    fi
    echo "Waiting for Bluetooth... ($i/10)"
  done
  echo "WARNING: Bluetooth init failed"
}

function launch {
  # Remove orphaned git lock if it exists on boot
  [ -f "$DIR/.git/index.lock" ] && rm -f $DIR/.git/index.lock

  # Check to see if there's a valid overlay-based update available. Conditions
  # are as follows:
  #
  # 1. The DIR init file has to exist, with a newer modtime than anything in
  #    the DIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.

  if [ -f "${DIR}/.overlay_init" ]; then
    find ${DIR}/.git -newer ${DIR}/.overlay_init | grep -q '.' 2> /dev/null
    if [ $? -eq 0 ]; then
      echo "${DIR} has been modified, skipping overlay update installation"
    else
      if [ -f "${STAGING_ROOT}/finalized/.overlay_consistent" ]; then
        if [ ! -d /data/safe_staging/old_openpilot ]; then
          echo "Valid overlay update found, installing"
          LAUNCHER_LOCATION="${BASH_SOURCE[0]}"

          mv $DIR /data/safe_staging/old_openpilot
          mv "${STAGING_ROOT}/finalized" $DIR
          cd $DIR

          echo "Restarting launch script ${LAUNCHER_LOCATION}"
          unset AGNOS_VERSION
          exec "${LAUNCHER_LOCATION}"
        else
          echo "openpilot backup found, not updating"
          # TODO: restore backup? This means the updater didn't start after swapping
        fi
      fi
    fi
  fi

  # handle pythonpath
  ln -sfn $(pwd) /data/pythonpath
  export PYTHONPATH="$PWD"

  # hardware specific init
  if [ -f /AGNOS ]; then
    agnos_init
    init_ble
  fi

  # write tmux scrollback to a file
  tmux capture-pane -pq -S-1000 > /tmp/launch_log

  # start manager
  cd system/manager
  if [ ! -f $DIR/prebuilt ]; then
    ./build.py
  fi
  ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
