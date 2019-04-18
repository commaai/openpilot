#!/bin/bash
set -e

docker build -t tmppilot -f Dockerfile.openpilot .

docker run --rm \
  -v "$(pwd)"/selfdrive/test/tests/plant/out:/data/openpilot/selfdrive/test/tests/plant/out \
  tmppilot /bin/sh -c 'cd /data/openpilot/selfdrive/test/tests/plant && OPTEST=1 ./test_longitudinal.py'
