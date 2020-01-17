#!/bin/bash
set -e

SETUP="cd /tmp/openpilot && "
RUN="docker run --shm-size 1G --rm tmppilot /bin/sh -c"

docker build -t tmppilot -f Dockerfile.openpilot .

$RUN "$SETUP cd /tmp/openpilot/selfdrive/test/ && ./test_fingerprints.py"
$RUN 'cd /tmp/openpilot/ && flake8 --select=F $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda" | grep -vi "^\./tools")'
$RUN 'cd /tmp/openpilot/ && pylint --disable=R,C,W $(find . -iname "*.py" | grep -vi "^\./pyextra.*" | grep -vi "^\./panda" | grep -vi "^\./tools"); exit $(($? & 3))'
$RUN "$SETUP python -m unittest discover common"
$RUN "$SETUP python -m unittest discover opendbc/can"
$RUN "$SETUP python -m unittest discover selfdrive/boardd"
$RUN "$SETUP python -m unittest discover selfdrive/controls"
$RUN "$SETUP python -m unittest discover selfdrive/loggerd"
$RUN "$SETUP cd /tmp/openpilot/selfdrive/test/longitudinal_maneuvers && OPTEST=1 ./test_longitudinal.py"
$RUN "$SETUP cd /tmp/openpilot/selfdrive/test/process_replay/ && ./test_processes.py"
$RUN "$SETUP mkdir -p /data/params && cd /tmp/openpilot/selfdrive/test/ && ./test_car_models.py"
