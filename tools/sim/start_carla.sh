#!/bin/bash

#Requires nvidia docker - https://github.com/NVIDIA/nvidia-docker
if ! $(apt list --installed | grep -q nvidia-container-toolkit); then
  if [ -z "$INSTALL" ]; then
    echo "Nvidia docker is required. Re-run with INSTALL=1 to automatically install."
    exit 0
  else
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    echo $distribution
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
  fi
fi

docker pull carlasim/carla:0.9.12

docker run \
  --rm \
  --net=host \
  -e DISPLAY= \
  -e SDL_VIDEODRIVER=offscreen \
  -it \
  --gpus all \
  carlasim/carla:0.9.12 \
  ./CarlaUE4.sh -opengl -nosound -quality-level=Epic
