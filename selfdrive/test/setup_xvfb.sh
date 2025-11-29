#!/usr/bin/env bash

# Sets up a virtual display for running map renderer and simulator without an X11 display

DISP_ID=99
export DISPLAY=:$DISP_ID

# Set resolution based on UI mode
if [ "${BIG:-1}" = "1" ]; then
  RESOLUTION="2160x1080x24"
else
  RESOLUTION="536x240x24"
fi

sudo Xvfb $DISPLAY -screen 0 $RESOLUTION 2>/dev/null &

# check for x11 socket for the specified display ID
while [ ! -S /tmp/.X11-unix/X$DISP_ID ]
do
  echo "Waiting for Xvfb..."
  sleep 1
done

touch ~/.Xauthority
export XDG_SESSION_TYPE="x11"
xset -q