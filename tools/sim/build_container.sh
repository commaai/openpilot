#!/bin/bash

docker pull docker.io/commaai/openpilot-sim:latest || true

if [ -n "$BUILD" ]; then
  cd ../../
  docker pull docker.io/commaai/openpilotci:latest || true
  docker build --cache-from commaai/openpilotci:latest -t commaai/openpilotci:latest -f Dockerfile.openpilot .
  docker build --cache-from commaai/openpilot-sim:latest -t commaai/openpilot-sim:latest -f tools/sim/Dockerfile.sim .
fi
