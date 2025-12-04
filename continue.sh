#!/usr/bin/env bash

echo -n 1 > /data/params/d/AdbEnabled
echo -n 1 > /data/params/d/SshEnabled
cp /usr/comma/setup_keys /data/params/d/GithubSshKeys
sudo systemctl stop power_monitor

sleep infinity
