#!/usr/bin/env bash

# Sets up a virtual display for running map renderer and simulator without an X11 display

if uname -r | grep -q "WSL2"; then
  DISP_ID=0 # WSLg uses display :0
else
  DISP_ID=99 # Standard Xvfb display
fi
export DISPLAY=:$DISP_ID

sudo Xvfb $DISPLAY -screen 0 2160x1080x24 2>/dev/null &

# check for x11 socket for the specified display ID
while [ ! -S /tmp/.X11-unix/X$DISP_ID ]
do
  echo "Waiting for Xvfb..."
  sleep 1
done

touch ~/.Xauthority
export XDG_SESSION_TYPE="x11"
xset -q