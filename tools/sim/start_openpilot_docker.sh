#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# expose X to the container
xhost +local:root

docker pull ghcr.io/commaai/openpilot-sim:latest

OPENPILOT_DIR="/openpilot"
if ! [[ -z "$MOUNT_OPENPILOT" ]]
then
  EXTRA_ARGS="-v $PWD/../..:/root/openpilot -e PYTHONPATH=/root/openpilot:$PYTHONPATH"
  OPENPILOT_DIR="/root/openpilot"
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
  ghcr.io/commaai/openpilot-sim:latest \
  /bin/bash -c "./tmux_script.sh $*"
