#!/bin/bash

if [ -n "$DEVELOP" ]; then
    echo "Develop Mode"
    echo "Run openpilot-sim with manager and bridge (local openpilot mounted as volume)"
    docker run  --shm-size 1G --rm --net=host -e PASSIVE=0 -e NOBOARD=1 -e CUDA_VISIBLE_DEVICES=  --volume="$HOME/openpilot:/tmp/openpilot" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --gpus all -e DISPLAY=$DISPLAY -it commaai/openpilot-sim:latest /bin/bash -c "cd tmp/openpilot && scons -j$(nproc) && screen -d -m python selfdrive/manager.py && screen tools/sim/bridge.py"
    
else 
    echo "Run openpilot-sim with manager and bridge"
    docker run  --shm-size 1G --rm --net=host -e PASSIVE=0 -e NOBOARD=1 -e CUDA_VISIBLE_DEVICES=  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --gpus all -e DISPLAY=$DISPLAY -it commaai/openpilot-sim:latest /bin/bash -c "cd tmp/openpilot && scons -j$(nproc) && screen -d -m python selfdrive/manager.py && screen tools/sim/bridge.py"
fi
