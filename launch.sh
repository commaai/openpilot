#!/usr/bin/bash

export PASSIVE="0"
export BASEDIR="/data/data/com.termux/files/one"

# Copy internal SSH keys
cp $HOME/one/ssh/authorized_keys /data/params/d/GithubSshKeys

exec ./launch_chffrplus.sh
