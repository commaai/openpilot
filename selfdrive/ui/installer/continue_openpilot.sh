#!/usr/bin/env bash

cd /data/openpilot
bash ./selfdrive/car/apply_default_car.sh 2>/dev/null || true
exec ./launch_openpilot.sh
