#!/bin/bash

# Requires nvidia docker - https://github.com/NVIDIA/nvidia-docker
docker pull carlasim/carla:0.9.7
docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.7
