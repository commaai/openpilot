#!/usr/bin/bash

export LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib
export HOME=/data/data/com.termux/files/home
export PATH=/usr/local/bin:/data/data/com.termux/files/usr/bin:/data/data/com.termux/files/usr/sbin:/data/data/com.termux/files/usr/bin/applets:/bin:/sbin:/vendor/bin:/system/sbin:/system/bin:/system/xbin:/data/data/com.termux/files/usr/bin/git
printf %s "1" > /data/params/d/DragonUpdating
cd /data/openpilot || exit
#git reset --hard @{u}
git clean -xdf
rm -fr /tmp/scons_cache/
find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
scons --clean
reboot