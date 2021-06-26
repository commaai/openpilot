#!/usr/bin/bash

function kill_process()
{
    proc_name=$1
    echo 'killing '$proc_name
    ps -ef | grep $proc_name | grep -v grep | awk '{print $1}' | xargs -r kill -9
}

function check_process()
{
    proc_name=$1
    ps -ef | grep $proc_name
}

kill_process thermald
kill_process manager.py
kill_process phone_control
kill_process manage_athenad
kill_process launch_chffrplus
kill_process athenad
kill_process can_bridge
kill_process loggerd
#kill_process start_op

check_process thermald
check_process manager.py
check_process phone_control

if [ -z "$1" ]; then
    echo ''
else
    kill_process start_op
fi