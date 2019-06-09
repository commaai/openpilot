#!/bin/bash
set -e

docker build -t tmppilot -f Dockerfile.openpilot .

docker run --rm \
  tmppilot /bin/sh -c 'cd /tmp/openpilot/selfdrive/test/ && ./test_fingerprints.py'
docker run --rm \
  tmppilot /bin/sh -c 'cd /tmp/openpilot/ && pyflakes $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda")'
docker run --rm \
  tmppilot /bin/sh -c 'cd /tmp/openpilot/ && pylint $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda"); exit $(($? & 3))'
docker run --rm \
  -v "$(pwd)"/selfdrive/test/tests/plant/out:/tmp/openpilot/selfdrive/test/tests/plant/out \
  tmppilot /bin/sh -c 'cd /tmp/openpilot/selfdrive/test/tests/plant && OPTEST=1 ./test_longitudinal.py'
