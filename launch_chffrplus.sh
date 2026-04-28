#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$DIR/launch_env.sh"

function agnos_init {
  # TODO: move this to agnos
  sudo rm -f /data/etc/NetworkManager/system-connections/*.nmmeta
  rm -f /data/scons_cache/config.lock

  # set success flag for current boot slot
  sudo abctl --set_success

  # udev does this, but sometimes we startup faster (downstream kernel only)
  for dev in /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0; do
    [ -e "$dev" ] && sudo chgrp gpu "$dev" && sudo chmod 660 "$dev"
  done

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

  PTH_FILE="/usr/local/venv/lib/python3.12/site-packages/system-packages.pth"
  if [ ! -f "$PTH_FILE" ]; then
    echo "Adding system packages to venv..."
    sudo mount -o remount,rw / 2>/dev/null || true
    echo '/usr/lib/python3/dist-packages' | sudo tee "$PTH_FILE" > /dev/null
  fi

  if pgrep -f "btattach.*ttyHS1" >/dev/null 2>&1 && hciconfig hci0 >/dev/null 2>&1; then
    echo "Bluetooth already running"
    return
  fi

  echo "Initializing Bluetooth..."
  sudo pkill -f btattach 2>/dev/null || true
  sudo hciconfig hci0 down 2>/dev/null || true
  sleep 1

  sudo btattach -B /dev/ttyHS1 -S 115200 &

  for i in $(seq 1 10); do
    sleep 1
    if hciconfig hci0 >/dev/null 2>&1; then
      echo "Bluetooth adapter found"
      sudo hciconfig hci0 down

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

  # hardware specific init (skip on ASIUS — Dragon has no A/B abctl, no
  # adsprpc-smd/ion/kgsl-3d0 chgrp targets, no agnos OTA)
  if [ -f /AGNOS ] && [ ! -f /ASIUS ]; then
    agnos_init
    init_ble
  fi

  # Build raylib PLATFORM_COMMA Python bindings on first boot (ASIUS only).
  # All source is baked in — no internet needed.
  if [ -f /ASIUS ] && [ ! -d "$DIR/third_party/raylib/wheel" ]; then
    echo "Building raylib PLATFORM_COMMA bindings..."
    RAYLIB_DIR="$DIR/third_party/raylib"
    VENV_PY=/usr/local/venv/bin/python3
    VENV_PIP=/usr/local/venv/bin/pip3
    cd "$RAYLIB_DIR/raylib_repo/src"
    make clean 2>/dev/null || true
    make -j$(nproc) PLATFORM=PLATFORM_COMMA RAYLIB_RELEASE_PATH="$RAYLIB_DIR/larch64"
    cp raylib.h raymath.h rlgl.h "$RAYLIB_DIR/include/"
    cd "$RAYLIB_DIR/raylib_python_repo"
    RAYLIB_PLATFORM=PLATFORM_COMMA \
      RAYLIB_INCLUDE_PATH="$RAYLIB_DIR/include" \
      RAYLIB_LIB_PATH="$RAYLIB_DIR/larch64" \
      $VENV_PY setup.py bdist_wheel
    mkdir -p "$RAYLIB_DIR/wheel"
    cp dist/*.whl "$RAYLIB_DIR/wheel/"
    sudo $VENV_PIP install --no-deps --force-reinstall "$RAYLIB_DIR/wheel/"*.whl
    cd "$DIR"
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
