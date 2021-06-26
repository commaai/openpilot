#!/usr/bin/bash

export LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib
export HOME=/data/data/com.termux/files/home
export PATH=/usr/local/bin:/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/sbin:/data/data/com.termux/files/usr/bin/applets:/bin:/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin:/data/data/com.termux/files/usr/bin/python
export PYTHONPATH=/data/openpilot
rm /data/openpilot/panda/board/obj/panda.bin
cd /data/openpilot/panda || exit ; pkill -f boardd ; python -c "from panda import Panda; Panda().flash()" && reboot