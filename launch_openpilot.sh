#!/usr/bin/bash
size=$(du -sb .git/index 2>/dev/null|awk '{print $1}')
echo $size|grep -E '^[0-9]+$' >/dev/null || size=0
if [ $size -le 1024 ];then
    rm .git/index 2>/dev/null
    git reset
fi

export PASSIVE="0"
exec ./launch_chffrplus.sh

