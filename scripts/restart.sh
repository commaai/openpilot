#!/bin/sh

ps ux|grep -v grep |grep '\{' |grep 'com\.'|awk '{print $1}'|xargs kill -9;  rm -rf /runonce; /comma.sh 
