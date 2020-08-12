#!/bin/bash

docker build -t openpilot -f Dockerfile.openpilot .
docker run  --shm-size 1G --rm --net=host -e CUDA_VISIBLE_DEVICES=  -e PASSIVE=0 -e NOBOARD=1 --volume="$HOME/openpilot/tools/sim:/tmp/openpilot/tools/sim-dev" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --gpus all -e DISPLAY=$DISPLAY -it openpilot /bin/bash

#Inside the docker container
# cd /tmp/openpilot/tools/sim
# ../../selfdrive/manager.py > /dev/null 2>&1 &
# ./bridge.py

