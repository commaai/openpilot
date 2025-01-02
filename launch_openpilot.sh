#!/usr/bin/env bash

if [ -z "$BASEDIR" ]; then
  BASEDIR="/data/openpilot"
fi
export PYTHONPATH="/data/openpilot"

pkill -f spinner
# launch spinner
echo "Building" | ./selfdrive/ui/spinner &
spinner_pid=$!

# build
scons selfdrive/ui/_notouch selfdrive/ui/_ui -j8
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
scons selfdrive/ui/_notouch selfdrive/ui/_ui -j8
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
