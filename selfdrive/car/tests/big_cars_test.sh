#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
BASEDIR=$(realpath "$SCRIPT_DIR/../../../")
cd $BASEDIR

MAX_EXAMPLES=300
INTERNAL_SEG_CNT=300
FILEREADER_CACHE=1
INTERNAL_SEG_LIST=selfdrive/car/tests/test_models_segs.txt

cd selfdrive/car/tests && pytest test_models.py test_car_interfaces.py
