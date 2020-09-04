#!/bin/bash -e

echo -e "\e[1;34m Getting libusb-1.0.23, compiling and installing for ARM64 cross-compilation... \e[0m"

#get libusb source
wget --tries=inf https://github.com/libusb/libusb/releases/download/v1.0.23/libusb-1.0.23.tar.bz2

tar xvf libusb-1.0.23.tar.bz2
pushd libusb-1.0.23

#configure it for aarch64
./configure --host=aarch64-linux-gnu --prefix=/usr/aarch64-linux-gnu 

#install it
sudo make install

popd

sudo rm -rf libusb-1.0.23*