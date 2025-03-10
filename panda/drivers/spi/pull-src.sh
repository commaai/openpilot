#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

rm -f spidev.c
wget https://raw.githubusercontent.com/commaai/agnos-kernel-sdm845/master/drivers/spi/spidev.c

# diff spidev.c spidev_panda.c > patch
# git diff --no-index spidev.c spidev_panda.c
patch -o spidev_panda.c spidev.c -i patch
