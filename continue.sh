#!/usr/bin/env bash

sudo chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
sudo chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0

echo -n 1 > /data/params/d/AdbEnabled
echo -n 1 > /data/params/d/SshEnabled
cp /usr/comma/setup_keys /data/params/d/GithubSshKeys
sudo systemctl stop power_monitor

sleep infinity
