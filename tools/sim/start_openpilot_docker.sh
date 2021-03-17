#!/bin/bash

# expose X to the container
xhost +local:root

docker pull ghcr.io/commaai/openpilot-sim:latest

if [[ -z "$MOUNT_OPENPILOT" ]]; then
  docker run --net=host\
    --name openpilot_client \
    --rm \
    -it \
    --gpus all \
    --device=/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --shm-size 1G \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    commaai/openpilot-sim:latest \
    /bin/bash -c "cd /openpilot/tools/sim && ./tmux_script.sh $*"
else
  docker run --net=host\
    --name openpilot_client \
    --rm \
    -it \
    --gpus all \
    --device=/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/openpilot:/root/openpilot \
    --shm-size 1G \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e PYTHONPATH=/root/openpilot:$PYTHONPATH \
    commaai/openpilot-sim:latest \
    /bin/bash -c "cd /root/openpilot/tools/sim && ./tmux_script.sh $*"
fi
