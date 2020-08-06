#!/bin/bash

# Requires nvidia docker - https://github.com/NVIDIA/nvidia-docker
docker pull carlasim/carla:0.8.2
docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.8.2
