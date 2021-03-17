#!/bin/bash

# expose X to the container
xhost +local:root

docker pull ghcr.io/commaai/openpilot-sim:latest

export EXTRA_ARGS=""
export OPENPILOT_DIR="/openpilot"
if ! [[ -z "$MOUNT_OPENPILOT" ]]
then
  export EXTRA_ARGS="-v $HOME/openpilot:/root/openpilot -e PYTHONPATH=/root/openpilot:$PYTHONPATH"
  export OPENPILOT_DIR="/root/openpilot"
fi

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
  -w "$OPENPILOT_DIR/tools/sim" \
  $EXTRA_ARGS \
  commaai/openpilot-sim:latest \
  /bin/bash -c "./tmux_script.sh $*"
