#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


#extract only the .egg file here (could maybe pipe from curl, maybe also directly from Dockerfile.sim)
FILE=CARLA_0.9.7.tar.gz
# rm -f $FILE
# curl -O http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/$FILE
# tar xvf ./$FILE PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg --strip-components=3
# rm -rf ./$FILE

cd $DIR/../../

docker pull commaai/openpilot-base:latest
docker build \
  --cache-from commaai/openpilot-sim:latest \
  -t commaai/openpilot-sim:latest \
  -f tools/sim/Dockerfile.sim .
