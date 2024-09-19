#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
BASEDIR=$(realpath "$SCRIPT_DIR/../../../")
cd $BASEDIR

export MAX_EXAMPLES=50
export FILEREADER_CACHE=1
export INTERNAL_SEG_LIST=selfdrive/car/tests/test_models_segs.txt

cd selfdrive/car/tests && pytest -n32 -k '(TOYOTA or LEXUS) and test_car_interface' test_models.py
