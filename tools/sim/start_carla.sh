#!/bin/bash
#echo "Your name is $1"
# Requires nvidia docker - https://github.com/NVIDIA/nvidia-docker
if ! $(apt list --installed | grep -q nvidia-container-toolkit); then
  read -p "Nvidia docker is required. Do you want to install it now? (y/n)";
  if [ "${REPLY}" == "y" ]; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    echo $distribution
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-docker2 # Also installs docker-ce and nvidia-container-toolkit
    read -p "Just installed docker? Set permanent docker permission for usergroup. (Run 'sudo systemctl restart docker' afterwards and then rerun start_carla.sh) (y/n)"
    if [ "${REPLY}" == "y" ]; then
        # Adding docker to current usergroup
        sudo groupadd docker
        sudo usermod -aG docker $USER
        newgrp docker
    fi
    sudo systemctl restart docker
  else
    exit
  fi
fi

docker pull carlasim/carla:0.9.12

docker run \
  --rm \
  --gpus all \
  --net=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -it \
  carlasim/carla:0.9.12 \
  /bin/bash ./CarlaUE4.sh -opengl -nosound -RenderOffScreen -benchmark -fps=20 -quality-level=High
