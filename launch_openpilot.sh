#!/usr/bin/env bash

ln -sfn $(pwd) /data/pythonpath
export PYTHONPATH="$PWD"

echo -n 1 > /data/params/d/AdbEnabled
echo -n 1 > /data/params/d/SshEnabled

# increase /tmp size for replay
sudo mount -o remount,size=4G /tmp

pkill -f spinner
# launch spinner
echo "Building" | ./system/ui/spinner.py &
spinner_pid=$!

# build
scons selfdrive/ui selfdrive/assets tools/replay msgq_repo common selfdrive/modeld system/camerad -j8 -j8
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
scons selfdrive/ui selfdrive/assets tools/replay msgq_repo common selfdrive/modeld system/camerad -j8 -j8
kill -9 $spinner_pid

# download videos
echo "Downloading videos" | ./system/ui/spinner.py &
spinner_pid=$!
./tools/scripts/download_replay_routes.py
kill -9 $spinner_pid

# launch notouch ui
NOTOUCH=1 ./system/manager/manager.py
