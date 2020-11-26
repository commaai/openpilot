#!/bin/bash

# expose X to the container
xhost +local:root
docker pull commaai/openpilot-sim:latest

docker run --net=host\
  --name openpilot_client \
  --rm \
  -it \
  --gpus all \
  --device=/dev/dri/  \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size 1G \
  -e DISPLAY=$DISPLAY \
  commaai/openpilot-sim:latest \
  /bin/bash -c "cd tools && cd sim && sh tmux_script.sh $*"
