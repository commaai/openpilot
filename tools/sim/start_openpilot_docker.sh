#!/bin/bash

# expose X to the container
xhost +local:root
#docker pull commaai/openpilot-sim:latest

docker run --net=host \
  --name openpilot_client \
  --rm \
  -it \
  -p 2000:2000 \
  --gpus all \
  --device=/dev/dri/  \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size 1G \
  -e DISPLAY=$DISPLAY \
  commaai/openpilot-sim:latest \
  /bin/bash -c "cd tools/sim && ./tmux_script.sh $*"
