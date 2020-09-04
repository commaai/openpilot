#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../../

docker pull commaai/openpilot-base:latest
docker build \
  --cache-from commaai/openpilot-sim:latest \
  -t commaai/openpilot-sim:latest \
  -f tools/sim/Dockerfile.sim .
