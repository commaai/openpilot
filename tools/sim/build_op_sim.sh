#!/bin/bash
echo "1. Pull openpilot-sim from dockerhub"
docker pull docker.io/commaai/openpilot-sim:latest || true
echo "2. Pull openpilotci from dockerhub"
docker pull docker.io/commaai/openpilotci:latest || true
pushd ../../
echo "3. Build openpilotci"
docker build --cache-from commaai/openpilotci:latest -t commaai/openpilotci:latest -f Dockerfile.openpilot .
echo "4. Build openpilot-sim"
docker build --cache-from commaai/openpilot-sim:latest -t commaai/openpilot-sim:latest -f tools/sim/Dockerfile.sim .
popd