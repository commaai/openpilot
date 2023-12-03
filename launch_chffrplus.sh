#!/usr/bin/bash

# TODO: delete this after 0.9.6 is in release

# migrate continue.sh and relaunch
sed -i 's/launch_chffrplus/launch_openpilot/g' /data/continue.sh
/data/continue.sh
