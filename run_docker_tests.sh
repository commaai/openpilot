#!/bin/bash
set -e

SETUP="cd /tmp/openpilot && make -C cereal && "

docker build -t tmppilot -f Dockerfile.openpilot .


docker run --rm tmppilot /bin/sh -c "$SETUP cd /tmp/openpilot/selfdrive/test/ && ./test_fingerprints.py"
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && flake8 --select=F $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda" | grep -vi "^\./tools")'
docker run --rm tmppilot /bin/sh -c 'cd /tmp/openpilot/ && pylint $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda" | grep -vi "^\./tools"); exit $(($? & 3))'
docker run --rm tmppilot /bin/sh -c "$SETUP python -m unittest discover common"
docker run --rm tmppilot /bin/sh -c "$SETUP make -C selfdrive/can -j4 && python -m unittest discover selfdrive/can"
docker run --rm tmppilot /bin/sh -c "$SETUP python -m unittest discover selfdrive/boardd"
docker run --rm tmppilot /bin/sh -c "$SETUP make -C selfdrive/can -j4 && python -m unittest discover selfdrive/controls"
docker run --rm tmppilot /bin/sh -c "$SETUP python -m unittest discover selfdrive/loggerd"
docker run --rm -v "$(pwd)"/selfdrive/test/longitudinal_maneuvers/out:/tmp/openpilot/selfdrive/test/longitudinal_maneuvers/out tmppilot /bin/sh -c "$SETUP make -C selfdrive/can -j4 && cd /tmp/openpilot/selfdrive/test/longitudinal_maneuvers && OPTEST=1 ./test_longitudinal.py"
docker run --rm tmppilot /bin/sh -c "$SETUP make -C selfdrive/can -j4 && cd /tmp/openpilot/selfdrive/test/process_replay/ && ./test_processes.py"
docker run --rm tmppilot /bin/sh -c "$SETUP make -C selfdrive/can -j4 && mkdir -p /data/params && cd /tmp/openpilot/selfdrive/test/ && ./test_car_models.py"
