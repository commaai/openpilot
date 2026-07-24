#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
BASEDIR=$(realpath "$SCRIPT_DIR/../../../../")
cd $BASEDIR

pytest -n logical --dist worksteal openpilot/selfdrive/car/tests/test_car_interfaces.py
