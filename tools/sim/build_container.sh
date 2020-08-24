#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../../

if [ -z "$(docker images -q commaai/openpilot-base:latest)" ]; then
  echo "pulling base docker image"
  docker pull commaai/openpilot-base:latest || true
fi
docker build \
  --cache-from commaai/openpilot-sim:latest \
  -t commaai/openpilot-sim:latest \
  -f tools/sim/Dockerfile.sim .
