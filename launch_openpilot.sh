#!/usr/bin/env bash

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

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="10.1"
fi
if [ -z "$BASEDIR" ]; then
  BASEDIR="/data/openpilot"
fi
export PYTHONPATH="/data/openpilot"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# hardware specific init
if [ -f /AGNOS ]; then
  agnos_init
fi

pkill -f spinner
# launch spinner
echo "Building" | ./selfdrive/ui/spinner &
spinner_pid=$!

# build
scons selfdrive/ui/_notouch selfdrive/ui/ui -j8
kill -9 $spinner_pid

# launch ui and let user set up ssh
./selfdrive/ui/ui

echo "Fetching updates" | ./selfdrive/ui/spinner &
spinner_pid=$!

# update
git fetch origin $(git rev-parse --abbrev-ref HEAD)
git reset --hard "@{u}"
git submodule update --init --recursive -f
kill -9 $spinner_pid

# build again after update
echo "Building" | ./selfdrive/ui/spinner &
spinner_pid=$!
scons selfdrive/ui/_notouch selfdrive/ui/ui -j8
kill -9 $spinner_pid

echo "Installing dependencies" | ./selfdrive/ui/spinner &
spinner_pid=$!

# install deps
sudo apt update
sudo apt-get install -y --no-install-recommends gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
kill -9 $spinner_pid

# download videos
echo "Downloading videos" | ./selfdrive/ui/spinner &
spinner_pid=$!
./selfdrive/assets/videos/download_videos.py
kill -9 $spinner_pid

# launch notouch ui
./selfdrive/ui/notouch
