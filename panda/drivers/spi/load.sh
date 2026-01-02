#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

make -j8

sudo su -c "echo spi0.0 > /sys/bus/spi/drivers/spidev/unbind" || true

sudo dmesg -C

#sudo rmmod -f spidev_panda
sudo rmmod spidev_panda || true
sudo insmod spidev_panda.ko

sudo su -c "echo 'file $DIR/spidev_panda.c +p' > /sys/kernel/debug/dynamic_debug/control"
sudo su -c "echo 'file $DIR/spi_panda.h +p' > /sys/kernel/debug/dynamic_debug/control"

sudo lsmod

echo "loaded"
ls -la /dev/spi*
sudo chmod 666 /dev/spi*
ipython -c "from panda import Panda; print(Panda.list())"
KERN=1 ipython -c "from panda import Panda; print(Panda.list())"
dmesg
