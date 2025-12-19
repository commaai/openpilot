#!/usr/bin/env bash

ln -sfn $(pwd) /data/pythonpath
export PYTHONPATH="$PWD"

pkill -f spinner
# launch spinner
echo "Building" | ./system/ui/spinner.py &
spinner_pid=$!

# build
scons selfdrive/ui tools/replay -j8
kill -9 $spinner_pid

# launch ui and let user set up ssh
./selfdrive/ui/ui.py

echo "Fetching updates" | ./system/ui/spinner.py &
spinner_pid=$!

# update
git fetch origin $(git rev-parse --abbrev-ref HEAD)
git reset --hard "@{u}"
git submodule update --init --recursive -f
kill -9 $spinner_pid

# build again after update
echo "Building" | ./system/ui/spinner.py &
spinner_pid=$!
scons selfdrive/ui tools/replay -j8
kill -9 $spinner_pid

echo "Installing dependencies" | ./system/ui/spinner.py &
spinner_pid=$!

# install deps
sudo apt update
sudo apt-get install -y --no-install-recommends gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
kill -9 $spinner_pid

# download videos
echo "Downloading videos" | ./system/ui/spinner.py &
spinner_pid=$!
./selfdrive/assets/videos/download_videos.py
kill -9 $spinner_pid

# launch notouch ui
#./selfdrive/ui/notouch
