#!/bin/bash
set -e

docker build -t tmppilot -f Dockerfile.openpilot .
docker run --rm \
  -v "$(pwd)"/selfdrive/test/plant/out:/tmp/openpilot/selfdrive/test/plant/out \
  tmppilot /bin/sh -c 'cd /tmp/openpilot/selfdrive/test/plant && ./runtest.sh'
