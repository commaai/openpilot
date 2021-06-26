#!/usr/bin/bash

cd /data/openpilot/panda ; pkill -f boardd ; PYTHONPATH=.. python -c "from panda import Panda; Panda().flash()"