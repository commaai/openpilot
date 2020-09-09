#!/bin/bash -e
echo -e "\e[1;34m Getting ffmpeg-4.3.1, compiling and installing for ARM64 cross-compilation... \e[0m"
wget --tries=inf https://ffmpeg.org/releases/ffmpeg-4.3.1.tar.gz

tar xvf ffmpeg-4.3.1.tar.gz
pushd ffmpeg-4.3.1

#configure it for aarch64
./configure --prefix=/usr/aarch64-linux-gnu --disable-x86asm

make
#install it
sudo make install

popd

sudo rm -rf ffmpeg-4.3.1*