#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
BASEDIR=$(realpath "$SCRIPT_DIR/../../../../")
cd $BASEDIR

export MAX_EXAMPLES=300
export INTERNAL_SEG_CNT=300
export INTERNAL_SEG_LIST=openpilot/selfdrive/car/tests/test_models_segs.txt

UNITTEST_JOBS=0 UNITTEST_LEVEL=class python -W error -m openpilot.common.parallel_runner \
  openpilot/selfdrive/car/tests/test_models.py openpilot/selfdrive/car/tests/test_car_interfaces.py
