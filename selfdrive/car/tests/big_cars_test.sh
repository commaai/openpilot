#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
BASEDIR=$(realpath "$SCRIPT_DIR/../../../")
cd $BASEDIR

export MAX_EXAMPLES=300
export INTERNAL_SEG_CNT=300
export FILEREADER_CACHE=1
export INTERNAL_SEG_LIST=selfdrive/car/tests/test_models_segs.txt

cd selfdrive/car/tests && pytest test_models.py test_car_interfaces.py
