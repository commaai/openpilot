#!/usr/bin/bash
cd /data/openpilot/selfdrive/visiond
if [ "$1" = "wiggly" ]; then
  ln -sf ./visiond-wiggly ./visiond
else
  ln -sf ./visiond-normal ./visiond
fi


tmux has-session -t comma
if [ $? == 0 ]
then
  tmux kill-session -t comma
fi
tmux new-session -d -s comma
tmux send-keys -t comma '/data/data/com.termux/files/continue.sh' C-m
  
