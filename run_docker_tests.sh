#!/bin/bash
set -e

# NOTE: this may be not an up-to-date list of tests and checks, see .github/workflows/test.yaml for the current list of tests

SETUP="cd /tmp/openpilot && "
BUILD="cd /tmp/openpilot && scons -c && scons -j$(nproc) && "
RUN="docker run --shm-size 1G --rm tmppilot /bin/bash -c"

docker build -t tmppilot -f Dockerfile.openpilot .

$RUN "$SETUP cd selfdrive/test/ && ./test_fingerprints.py"
$RUN "$SETUP git init && git add -A && pre-commit run --all"
$RUN "$BUILD python -m unittest discover common && \
             python -m unittest discover opendbc/can && \
             python -m unittest discover selfdrive/boardd && \
             python -m unittest discover selfdrive/controls && \
             python -m unittest discover selfdrive/loggerd && \
             python -m unittest discover selfdrive/car && \
             python -m unittest discover selfdrive/locationd && \
             python -m unittest discover selfdrive/athena"
$RUN "$BUILD cd /tmp/openpilot/selfdrive/test/longitudinal_maneuvers && OPTEST=1 ./test_longitudinal.py"
$RUN "$BUILD cd /tmp/openpilot/selfdrive/test/process_replay/ && ./test_processes.py"
$RUN "$BUILD mkdir -p /data/params && cd /tmp/openpilot/selfdrive/test/ && ./test_car_models.py"
