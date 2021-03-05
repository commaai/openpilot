#!/bin/bash

# expose X to the container
sudo xhost +local:root

docker run --net=host\
  --name openpilot_client \
  --rm \
  -it \
  --privileged \
  --gpus all \
  --device=/dev/dri \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /run/user/1000:/run/user/1000 \
  --shm-size 1G \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  commaai/openpilot-sim:latest \
  /bin/bash -c "cd /openpilot/tools && cd sim && sh tmux_script.sh $*"
