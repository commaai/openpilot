#!/usr/bin/env bash

sudo mount --bind / /base
mount -t overlay overlay -o lowerdir=/base,upperdir=/upper,workdir=/work /newroot
rm -f /newroot/etc/resolv.conf
touch /newroot/etc/resolv.conf
cat /etc/resolv.conf > /newroot/etc/resolv.conf



mkdir -p /newroot/old
cd /newroot
pivot_root . old

mount -t proc proc /proc
mount -t devtmpfs devtmpfs /dev
mkdir -p /dev/pts
mount -t devpts devpts /dev/pts
mount -t proc proc /proc
mount -t sysfs sysfs /sys



mount

touch /root_committed
sudo -u runner /home/runner/work/openpilot/openpilot/selfdrive/test/build.sh
ec=$?

echo "hello"

#sudo umount /newroot
#sudo umount /base

exit $ec
