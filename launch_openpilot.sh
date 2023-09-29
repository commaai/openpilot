#!/usr/bin/bash

echo -n 1 > /data/params/d/WheeledBody

echo -n 1 > /data/params/d/SshEnabled
echo -n 1 > /data/params/d/DisableUpdates
# cp /usr/comma/setup_keys /data/params/d/GithubSshKeys

export PASSIVE="0"
exec ./launch_chffrplus.sh
