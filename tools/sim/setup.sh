#!/bin/bash

# install nvidia docker - https://github.com/NVIDIA/nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker


# TODO: make non-interactive
# install opencl


# TODO: clean this up
# install carla python api
FILE=CARLA_0.9.7.tar.gz
curl -O http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/$FILE
tar xvf $FILE
easy_install PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg || trueFILE=CARLA_0.9.7.tar.gz
if [ ! -f $FILE ]; then
  curl -O http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/$FILE
fi
if [ ! -d carla ]; then
  rm -rf carla_tmp
  mkdir -p carla_tmp
  cd carla_tmp
  tar xvf ../$FILE
  easy_install PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg || true
  cd ../
  mv carla_tmp carla
fi


